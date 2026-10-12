"""Dữ liệu: đọc HDF5 -> cache đã crop (memmap) -> Dataset chunk -> augmentation trên GPU.

Chuẩn bị cache (chạy 1 lần):
    python data.py --raw_dir data/raw --cache_dir data/cache --crop_xyxy "[104,48,536,480]"

Các rủi ro được xử lý ở file này:
  1. Crop vùng làm việc TRƯỚC khi resize (preprocess_frame).
  2. Không cache feature backbone: cache ảnh crop 256px, augment online (gpu_preprocess).
  3. Thống kê chuẩn hoá Δq theo từng (bước, khớp) (compute_stats).
  4. Ghi nhận tay đang hoạt động cho từng episode (detect_active_arm) để làm mask loss.
  5. Depth: đổi ra mét, clip [min, max], mask pixel lỗi, nội suy chuẩn hoá theo mask.
  6. Tách val theo episode, lấy mẫu cân bằng giữa hai tay.
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from config import PREP_KEYS, Config, parse_config


# =====================================================================
# Đọc HDF5
# =====================================================================
def h5_datasets(f: h5py.File) -> dict:
    out = {}

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            out[name] = obj

    f.visititems(visit)
    return out


def print_h5_tree(f: h5py.File):
    for name, ds in h5_datasets(f).items():
        print(f"  {name:50s} shape={ds.shape} dtype={ds.dtype}")


def autodetect_keys(f: h5py.File, cfg: Config) -> dict:
    """Trả về {'rgb','depth','cmd','state'} -> tên dataset trong file."""
    dsets = h5_datasets(f)
    D = cfg.action_dim

    def is_img(ds):
        return ds.ndim == 1 or (ds.ndim == 4 and ds.shape[-1] == 3)

    def is_depth(ds):
        return ds.ndim in (1, 3) or (ds.ndim == 4 and ds.shape[-1] == 1)

    def is_vec(ds):
        return ds.ndim == 2 and ds.shape[-1] == D

    rules = {
        "rgb": (("rgb", "color", "colour", "image", "img", "cam"), ("depth",), is_img),
        "depth": (("depth",), (), is_depth),
        "cmd": (("command", "cmd", "action"), (), is_vec),
        "state": (("joint_state", "state", "qpos", "position"),
                  ("command", "cmd", "action", "velocit", "effort", "torque"), is_vec),
    }
    given = {"rgb": cfg.key_rgb, "depth": cfg.key_depth, "cmd": cfg.key_cmd, "state": cfg.key_state}
    found = {}
    for role, (must, avoid, cond) in rules.items():
        if given[role]:
            if given[role] not in f:
                raise KeyError(f"Không có key '{given[role]}' trong {f.filename}")
            found[role] = given[role]
            continue
        cands = [n for n, ds in dsets.items()
                 if any(m in n.lower() for m in must)
                 and not any(a in n.lower() for a in avoid)
                 and n not in found.values() and cond(ds)]
        if not cands:
            print(f"[data] Không tự dò được '{role}' trong {f.filename}. Cấu trúc file:")
            print_h5_tree(f)
            raise KeyError(f"Hãy đặt --key_{role} <tên dataset>")
        # ưu tiên theo thứ tự từ khoá (vd. "command" trước "action"), rồi tên ngắn hơn
        rank = lambda n: (min(i for i, m in enumerate(must) if m in n.lower()), len(n))
        found[role] = sorted(cands, key=rank)[0]
    return found


def valid_start(cmd: np.ndarray, state: np.ndarray, name: str = "") -> int:
    """Frame đầu tiên có cmd và state hữu hạn. Lệnh teleop là NaN ở đầu episode (trước khi bắt đầu
    gửi lệnh) -> bỏ đoạn đó. NaN ở giữa episode thì báo lỗi thay vì lặng lẽ học sai."""
    ok = np.isfinite(cmd).all(1) & np.isfinite(state).all(1)
    if not ok.any():
        raise ValueError(f"{name}: toàn bộ cmd/state là NaN")
    s = int(np.argmax(ok))
    if not ok[s:].all():
        raise ValueError(f"{name}: có NaN giữa episode tại frame {(np.where(~ok[s:])[0] + s)[:5]}")
    return s


def _decode(buf, flags):
    arr = np.frombuffer(bytes(buf) if not isinstance(buf, np.ndarray) else buf.tobytes(), np.uint8)
    img = cv2.imdecode(arr, flags)
    if img is None:
        raise ValueError("Không giải mã được ảnh nén trong HDF5")
    return img


def read_block(ds: h5py.Dataset, idx: np.ndarray, kind: str) -> list:
    """Đọc các frame tại chỉ số idx (tăng dần). kind: 'rgb' | 'depth'."""
    if ds.ndim == 1:  # ảnh nén (JPEG/PNG) dạng bytes
        flags = cv2.IMREAD_COLOR if kind == "rgb" else cv2.IMREAD_UNCHANGED
        return [_decode(ds[int(i)], flags) for i in idx]
    block = ds[int(idx[0]): int(idx[-1]) + 1]
    block = block[idx - idx[0]]
    return [b for b in block]


def to_rgb(img: np.ndarray, cfg: Config) -> np.ndarray:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if cfg.rgb_is_bgr:
        img = img[..., ::-1]
    return np.ascontiguousarray(img.astype(np.uint8))


def to_meters(depth: np.ndarray, cfg: Config) -> np.ndarray:
    d = np.asarray(depth)
    if d.ndim == 3:
        d = d[..., 0]
    if np.issubdtype(d.dtype, np.integer):
        d = d.astype(np.float32) * cfg.depth_unit_m
    else:
        d = d.astype(np.float32)
        valid = d[np.isfinite(d) & (d > 0)]
        if valid.size and np.median(valid) > 20:  # float nhưng tính bằng mm
            d = d * 0.001
    d[~np.isfinite(d)] = 0.0
    return d


# =====================================================================
# RỦI RO 1: crop vùng làm việc trước khi resize
# =====================================================================
def crop_box(cfg: Config, w: int, h: int, ref_w: int, ref_h: int):
    """crop_xyxy định nghĩa trên ảnh RGB (ref_w x ref_h); quy đổi sang ảnh (w x h)."""
    if cfg.crop_xyxy is None:
        return 0, 0, w, h
    x0, y0, x1, y1 = cfg.crop_xyxy
    sx, sy = w / ref_w, h / ref_h
    x0, x1 = int(round(x0 * sx)), int(round(x1 * sx))
    y0, y1 = int(round(y0 * sy)), int(round(y1 * sy))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        raise ValueError(f"crop_xyxy {cfg.crop_xyxy} nằm ngoài ảnh {ref_w}x{ref_h}")
    return x0, y0, x1, y1


def preprocess_frame(rgb: np.ndarray, depth_m: np.ndarray, cfg: Config):
    """Ảnh gốc -> (RGB crop cache_rgb_size uint8, depth crop cache_depth_size float16 mét).

    Dùng chung cho tạo cache và chạy thật trên robot để đảm bảo xử lý giống hệt nhau.
    """
    H, W = rgb.shape[:2]
    x0, y0, x1, y1 = crop_box(cfg, W, H, W, H)
    r = cv2.resize(rgb[y0:y1, x0:x1], (cfg.cache_rgb_size, cfg.cache_rgb_size),
                   interpolation=cv2.INTER_AREA)
    h, w = depth_m.shape[:2]
    dx0, dy0, dx1, dy1 = crop_box(cfg, w, h, W, H)
    # NEAREST để không trộn pixel lỗi (=0) với pixel hợp lệ
    d = cv2.resize(depth_m[dy0:dy1, dx0:dx1], (cfg.cache_depth_size, cfg.cache_depth_size),
                   interpolation=cv2.INTER_NEAREST)
    return r, d.astype(np.float16)


# =====================================================================
# RỦI RO 4: xác định tay đang làm việc của mỗi episode
# =====================================================================
def detect_active_arm(cmd: np.ndarray, cfg: Config):
    motion = []
    for a in (0, 1):
        j = list(cfg.arm_joints[a])
        motion.append(float((cmd[:, j].max(0) - cmd[:, j].min(0)).sum()))
    arm = int(np.argmax(motion))
    ratio = motion[arm] / max(motion[1 - arm], 1e-6)
    return arm, motion, ratio


# =====================================================================
# RỦI RO 3: thống kê chuẩn hoá theo từng (bước, khớp)
# =====================================================================
def make_action(cmd_chunk: np.ndarray, state: np.ndarray, cfg: Config) -> np.ndarray:
    """cmd_chunk (..., K, D), state (..., D) -> action: Δq cho khớp tay, tuyệt đối cho kẹp."""
    a = cmd_chunk.astype(np.float32).copy()
    dc = cfg.delta_cols
    a[..., dc] -= state[..., None, :][..., dc]
    return a


def compute_stats(cmd, state, episodes, cfg: Config) -> dict:
    K, D = cfg.chunk, cfg.action_dim
    s1 = np.zeros((K, D)); s2 = np.zeros((K, D)); n = np.zeros((K, D))
    st_all = []
    for ep in episodes:
        s, e, arm = ep["start"], ep["end"], ep["arm"]
        T = e - s
        c, st = np.asarray(cmd[s:e], np.float64), np.asarray(state[s:e], np.float64)
        st_all.append(st)
        idx = np.arange(T)[:, None] + np.arange(K)[None]
        valid = idx < T
        idx = np.minimum(idx, T - 1)
        a = make_action(c[idx], st, cfg).astype(np.float64)     # (T, K, D)
        col = np.zeros(D, bool)
        col[cfg.arm_cols(arm)] = True                            # chỉ tính tay đang làm việc
        m = (valid[..., None] & col[None, None]).astype(np.float64)
        s1 += (a * m).sum(0); s2 += (a * a * m).sum(0); n += m.sum(0)
    if (n == 0).any():
        print("[data] CẢNH BÁO: có cột chưa từng hoạt động trong tập train -> dùng mean=0, std=1")
    mean = np.where(n > 0, s1 / np.maximum(n, 1), 0.0)
    var = np.where(n > 0, s2 / np.maximum(n, 1) - mean ** 2, 1.0)
    std = np.maximum(np.sqrt(np.maximum(var, 0)), 1e-3)
    st_all = np.concatenate(st_all)
    return {
        "action_mean": mean.astype(np.float32), "action_std": std.astype(np.float32),
        "state_mean": st_all.mean(0).astype(np.float32),
        "state_std": np.maximum(st_all.std(0), 1e-3).astype(np.float32),
    }


# =====================================================================
# Tạo cache
# =====================================================================
def check_norm_range(cmd, state, episodes, stats, cfg: Config):
    """In phân vị 99.9% của |giá trị chuẩn hoá| cho tay làm việc và tay đứng yên.
    Tay đứng yên dùng thống kê của tay làm việc; nếu robot võng cố định (cmd != state khi đứng yên)
    giá trị có thể rất lớn -> cảnh báo."""
    K = cfg.chunk
    act_v, idle_v = [], []
    for ep in episodes:
        s, e, arm = ep["start"], ep["end"], ep["arm"]
        T = e - s
        idx = np.minimum(np.arange(T)[:, None] + np.arange(K)[None], T - 1)
        a = make_action(np.asarray(cmd[s:e])[idx], np.asarray(state[s:e]), cfg)
        z = np.abs((a - stats["action_mean"]) / stats["action_std"])
        act_v.append(z[..., cfg.arm_cols(arm)].ravel())
        idle_v.append(z[..., cfg.arm_cols(1 - arm)].ravel())
    pa = np.percentile(np.concatenate(act_v), 99.9)
    pi = np.percentile(np.concatenate(idle_v), 99.9)
    print(f"[data] |action chuẩn hoá| p99.9: tay làm việc {pa:.2f}, tay đứng yên {pi:.2f}")
    if pi > 8:
        print("[data] CẢNH BÁO: tay đứng yên có giá trị chuẩn hoá lớn (độ võng cmd-state?). "
              "Cân nhắc giảm --inactive_weight hoặc kiểm tra joint_commands của tay không dùng.")


def find_h5(raw_dir):
    files = []
    for ext in ("h5", "hdf5"):
        files += glob.glob(os.path.join(raw_dir, "**", f"*.{ext}"), recursive=True)
    return sorted(files)


def split_episodes(episodes: list, val_frac: float, seed: int) -> list:
    """RỦI RO 6: tách val theo episode (không theo frame), phân tầng theo tay. Ghi e["split"]."""
    rng = np.random.default_rng(seed)
    for a in (0, 1):
        ids = [i for i, e in enumerate(episodes) if e["arm"] == a]
        rng.shuffle(ids)
        n_val = max(1, int(round(val_frac * len(ids)))) if len(ids) >= 4 and val_frac > 0 else 0
        for k, i in enumerate(ids):
            episodes[i]["split"] = "val" if k < n_val else "train"
    for e in episodes:
        e.setdefault("split", "train")
    return episodes


def prepare(cfg: Config):
    files = find_h5(cfg.raw_dir)
    if not files and cfg.hf_repo:
        from huggingface_hub import snapshot_download
        print(f"[data] Tải {cfg.hf_repo} về {cfg.raw_dir} ...")
        snapshot_download(repo_id=cfg.hf_repo, repo_type="dataset", local_dir=cfg.raw_dir)
        files = find_h5(cfg.raw_dir)
    if not files:
        sys.exit(f"[data] Không tìm thấy file HDF5 trong {cfg.raw_dir}")
    os.makedirs(cfg.cache_dir, exist_ok=True)

    # ---- lượt 1: dò key, đếm frame
    plan, total = [], 0
    for fp in files:
        with h5py.File(fp, "r") as f:
            keys = autodetect_keys(f, cfg)
            lens = {r: f[k].shape[0] for r, k in keys.items()}
            T_raw = min(lens.values())
            if len(set(lens.values())) > 1:
                print(f"[data] {os.path.basename(fp)}: độ dài lệch {lens}, cắt về {T_raw}")
            s0 = valid_start(f[keys["cmd"]][:T_raw], f[keys["state"]][:T_raw], os.path.basename(fp))
            idx = np.arange(s0, T_raw, cfg.frame_stride)
            plan.append((fp, keys, idx))
            total += len(idx)
    print(f"[data] {len(files)} episode, {total} frame. Key: {plan[0][1]}")

    R, Dd, D = cfg.cache_rgb_size, cfg.cache_depth_size, cfg.action_dim
    mm = lambda name, dt, shp: np.lib.format.open_memmap(
        os.path.join(cfg.cache_dir, name), mode="w+", dtype=dt, shape=shp)
    rgb_mm = mm("rgb.npy", np.uint8, (total, R, R, 3))
    dep_mm = mm("depth.npy", np.float16, (total, Dd, Dd))
    cmd_mm = mm("cmd.npy", np.float32, (total, D))
    st_mm = mm("state.npy", np.float32, (total, D))

    # ---- lượt 2: ghi cache
    episodes, g = [], 0
    for ei, (fp, keys, idx) in enumerate(plan):
        with h5py.File(fp, "r") as f:
            cmd = np.asarray(f[keys["cmd"]][:], np.float32)[idx]
            st = np.asarray(f[keys["state"]][:], np.float32)[idx]
            if np.array_equal(cmd, st):
                print(f"[data] CẢNH BÁO {os.path.basename(fp)}: '{keys['cmd']}' trùng hệt '{keys['state']}' "
                      "-> đây không phải lệnh, hãy đặt --key_cmd")
            for b in range(0, len(idx), 64):
                bi = idx[b:b + 64]
                rgbs = read_block(f[keys["rgb"]], bi, "rgb")
                deps = read_block(f[keys["depth"]], bi, "depth")
                for j, (r, d) in enumerate(zip(rgbs, deps)):
                    rc, dc = preprocess_frame(to_rgb(r, cfg), to_meters(d, cfg), cfg)
                    rgb_mm[g + b + j] = rc
                    dep_mm[g + b + j] = dc
        T = len(idx)
        cmd_mm[g:g + T] = cmd
        st_mm[g:g + T] = st
        arm, motion, ratio = detect_active_arm(cmd, cfg)
        flag = "  <-- MƠ HỒ, kiểm tra lại bố cục khớp" if ratio < 2 else ""
        print(f"[data] {ei:3d} {os.path.basename(fp):40s} T={T:5d} (bỏ {idx[0]} frame NaN đầu) tay={cfg.arm_names[arm]} "
              f"chuyển động={motion[0]:.2f}/{motion[1]:.2f}{flag}")
        episodes.append({"file": os.path.basename(fp), "start": g, "end": g + T, "arm": arm})
        g += T
    for m in (rgb_mm, dep_mm, cmd_mm, st_mm):
        m.flush()

    split_episodes(episodes, cfg.val_frac, cfg.seed)
    train_eps = [e for e in episodes if e["split"] == "train"]
    stats = compute_stats(cmd_mm, st_mm, train_eps, cfg)
    np.savez(os.path.join(cfg.cache_dir, "stats.npz"), **stats)
    check_norm_range(cmd_mm, st_mm, train_eps, stats, cfg)

    meta = {"cfg": {k: getattr(cfg, k) for k in PREP_KEYS}, "episodes": episodes, "total": total}
    meta["cfg"].update({f"key_{r}": k for r, k in plan[0][1].items()})
    with open(os.path.join(cfg.cache_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    for a in (0, 1):
        tr = sum(e["arm"] == a and e["split"] == "train" for e in episodes)
        va = sum(e["arm"] == a and e["split"] == "val" for e in episodes)
        print(f"[data] {cfg.arm_names[a]}: {tr} train / {va} val episode")
    print("[data] std Δq bước 0 vs bước cuối (tay 0):",
          np.round(stats["action_std"][0, list(cfg.arm_joints[0])], 4),
          np.round(stats["action_std"][-1, list(cfg.arm_joints[0])], 4))
    print(f"[data] Xong -> {cfg.cache_dir}")


def load_prep_cfg(cfg: Config, cache_dir: str) -> tuple:
    """Ghi đè cfg bằng các trường lúc tạo cache để train/infer khớp với dữ liệu."""
    with open(os.path.join(cache_dir, "meta.json")) as fh:
        meta = json.load(fh)
    merged = Config.from_dict({**cfg.to_dict(), **meta["cfg"]})
    merged.cache_dir = cache_dir
    stats = dict(np.load(os.path.join(cache_dir, "stats.npz")))
    return merged, meta, stats


# =====================================================================
# Dataset
# =====================================================================
class ChunkDataset(Dataset):
    def __init__(self, cfg: Config, meta: dict, split: str):
        self.cfg = cfg
        self.eps = [e for e in meta["episodes"] if e["split"] == split]
        index = []
        for e in self.eps:
            for g in range(e["start"], e["end"]):
                index.append((g, e["end"], e["arm"]))
        self.index = np.array(index, dtype=np.int64).reshape(-1, 3)
        self._mm = None

    def _open(self):
        if self._mm is None:
            c = self.cfg.cache_dir
            self._mm = {k: np.load(os.path.join(c, f"{k}.npy"), mmap_mode="r")
                        for k in ("rgb", "depth", "cmd", "state")}
        return self._mm

    def __len__(self):
        return len(self.index)

    def sample_weights(self) -> torch.Tensor:
        arms = self.index[:, 2]
        cnt = np.bincount(arms, minlength=2).astype(np.float64)
        return torch.as_tensor(1.0 / np.maximum(cnt[arms], 1))

    def __getitem__(self, i):
        mm = self._open()
        g, end, arm = (int(x) for x in self.index[i])
        K = self.cfg.chunk
        idx = g + np.arange(K)
        pad = idx >= end
        idx = np.minimum(idx, end - 1)
        state = np.array(mm["state"][g], np.float32)
        action = make_action(np.asarray(mm["cmd"][idx]), state, self.cfg)
        return {
            "rgb": torch.from_numpy(np.array(mm["rgb"][g])),              # (R,R,3) u8
            "depth": torch.from_numpy(np.array(mm["depth"][g], np.float32)),        # (Dd,Dd) m
            "state": torch.from_numpy(state),
            "action": torch.from_numpy(action),                                        # (K,D) thô
            "pad": torch.from_numpy(pad),
            "arm": torch.tensor(arm),
        }


# =====================================================================
# RỦI RO 2 + 5: augmentation / tiền xử lý trên GPU (không cache feature)
# =====================================================================
def _boxes(B, train, cfg, device):
    if train:
        lo, hi = cfg.crop_scale_train
        s = torch.empty(B, device=device).uniform_(lo, hi)
        cx = s / 2 + torch.rand(B, device=device) * (1 - s)
        cy = s / 2 + torch.rand(B, device=device) * (1 - s)
    else:
        s = torch.full((B,), cfg.crop_scale_eval, device=device)
        cx = cy = torch.full((B,), 0.5, device=device)
    return s, cx, cy


def _roi(x, s, cx, cy, out, sampling_ratio):
    from torchvision.ops import roi_align
    B, _, H, W = x.shape
    bi = torch.arange(B, device=x.device, dtype=x.dtype)
    boxes = torch.stack([bi, (cx - s / 2) * W, (cy - s / 2) * H,
                         (cx + s / 2) * W, (cy + s / 2) * H], 1).to(x.dtype)
    return roi_align(x, boxes, (out, out), 1.0, sampling_ratio, aligned=True)


def _color_jitter(x, cfg):
    B = x.shape[0]
    u = lambda a: 1 + (torch.rand(B, 1, 1, 1, device=x.device) * 2 - 1) * a
    w = torch.tensor([0.299, 0.587, 0.114], device=x.device).view(1, 3, 1, 1)
    x = x * u(cfg.aug_brightness)
    m = (x * w).sum(1, keepdim=True).mean((2, 3), keepdim=True)
    x = (x - m) * u(cfg.aug_contrast) + m
    gray = (x * w).sum(1, keepdim=True)
    x = (x - gray) * u(cfg.aug_saturation) + gray
    x = x + torch.randn_like(x) * cfg.aug_rgb_noise
    return x.clamp(0, 1)


def _depth_holes(valid, cfg, max_rects=3):
    B, _, S, _ = valid.shape
    dev = valid.device
    on = (torch.rand(B, max_rects, device=dev) < cfg.depth_hole_prob)
    size = torch.randint(S // 16, S // 4 + 1, (B, max_rects, 2), device=dev)
    x0 = (torch.rand(B, max_rects, device=dev) * (S - size[..., 0])).long()
    y0 = (torch.rand(B, max_rects, device=dev) * (S - size[..., 1])).long()
    ar = torch.arange(S, device=dev)
    inx = (ar.view(1, 1, 1, S) >= x0[..., None, None]) & (ar.view(1, 1, 1, S) < (x0 + size[..., 0])[..., None, None])
    iny = (ar.view(1, 1, S, 1) >= y0[..., None, None]) & (ar.view(1, 1, S, 1) < (y0 + size[..., 1])[..., None, None])
    hole = (inx & iny & on[..., None, None]).any(1, keepdim=True)
    return valid * (~hole).float()


def gpu_preprocess(batch: dict, cfg: Config, train: bool):
    """-> rgb (B,3,img,img) trong [0,1], depth (B,2,ds,ds) = [độ sâu chuẩn hoá, mask hợp lệ].

    RGB và depth dùng CHUNG một hộp crop ngẫu nhiên nên vẫn thẳng hàng.
    """
    rgb = batch["rgb"].permute(0, 3, 1, 2).float() / 255.0
    d = batch["depth"].float().unsqueeze(1)
    B, dev = rgb.shape[0], rgb.device
    s, cx, cy = _boxes(B, train, cfg, dev)

    rgb = _roi(rgb, s, cx, cy, cfg.img_size, sampling_ratio=2)
    if train:
        rgb = _color_jitter(rgb, cfg)

    lo, hi = cfg.depth_min_m, cfg.depth_max_m
    valid = ((d >= lo) & (d <= hi)).float()
    if train and cfg.depth_noise_m > 0:
        d = d + torch.randn_like(d) * cfg.depth_noise_m * valid
    d = d.clamp(lo, hi) * valid
    # nội suy chuẩn hoá theo mask: pixel lỗi không kéo lệch pixel hợp lệ
    num_den = _roi(torch.cat([d, valid], 1), s, cx, cy, cfg.depth_size, sampling_ratio=1)
    num, den = num_den[:, :1], num_den[:, 1:]
    v = (den > 0.5).float()
    if train and cfg.depth_hole_prob > 0:
        v = _depth_holes(v, cfg)
    dn = ((num / den.clamp_min(1e-6) - lo) / (hi - lo)).clamp(0, 1) * v
    return rgb, torch.cat([dn, v], 1)


if __name__ == "__main__":
    prepare(parse_config(description="Tạo cache dữ liệu OpenArm"))
