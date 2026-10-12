"""Cấu hình chung cho VA flow-matching policy (OpenArm, 16 DoF).

Mọi trường đều ghi đè được từ dòng lệnh:  --batch_size 32  --crop_xyxy "[100,40,540,480]"
hoặc nạp từ file JSON:                    --config my_cfg.json
"""
from __future__ import annotations

import argparse
import inspect
import json
import re
import typing
from dataclasses import asdict, dataclass, fields
from typing import Optional, Tuple


@dataclass
class Config:
    # ======================= DỮ LIỆU THÔ =======================
    raw_dir: str = "data/raw"                 # thư mục chứa *.h5 / *.hdf5 (đệ quy)
    cache_dir: str = "data/cache"             # cache đã crop + resize (memmap)
    hf_repo: Optional[str] = "Tuyen062004/data_openarm_8_10"  # tải về raw_dir nếu raw_dir trống (None = không tải)
    # Key HDF5. None = tự dò (xem: python tools.py inspect <file.h5>)
    key_rgb: Optional[str] = None
    key_depth: Optional[str] = None
    key_cmd: Optional[str] = None             # joint_commands = vị trí mục tiêu (action)
    key_state: Optional[str] = None           # joint_states = vị trí hiện tại (qpos)
    rgb_is_bgr: bool = False                  # True nếu mảng ảnh lưu theo BGR (xem preview-crop)
    depth_unit_m: float = 0.001               # depth nguyên (uint16) tính bằng mm -> m
    frame_stride: int = 1                     # >1 nếu dữ liệu ghi nhanh hơn `hz`

    # ======================= BỐ CỤC KHỚP =======================
    # 16 = 2 tay x (7 khớp + 1 kẹp). Kiểm tra bằng: python tools.py joints
    action_dim: int = 16
    arm_joints: Tuple[Tuple[int, ...], Tuple[int, ...]] = (
        (0, 1, 2, 3, 4, 5, 6),
        (8, 9, 10, 11, 12, 13, 14),
    )
    grippers: Tuple[int, int] = (7, 15)
    arm_names: Tuple[str, str] = ("arm0", "arm1")

    # ============ RỦI RO 1: CROP VÙNG LÀM VIỆC TRƯỚC KHI RESIZE ============
    # Toạ độ pixel (x0, y0, x1, y1) trên ảnh RGB gốc. None = cả khung hình.
    # Chỉnh bằng: python tools.py preview-crop --crop_xyxy "[x0,y0,x1,y1]"
    crop_xyxy: Optional[Tuple[int, int, int, int]] = (104, 48, 536, 480)
    cache_rgb_size: int = 256                 # vùng crop lưu ở 256 để còn chỗ random-crop
    cache_depth_size: int = 160
    img_size: int = 224                       # đầu vào backbone -> (224/14)^2 = 256 token
    depth_size: int = 128                     # đầu vào CNN depth -> (128/16)^2 = 64 token
    crop_scale_train: Tuple[float, float] = (0.84, 1.0)  # cạnh crop ngẫu nhiên (tỉ lệ)
    crop_scale_eval: float = 0.92                         # crop giữa cố định khi eval/chạy thật

    # ============ RỦI RO 5: DEPTH REALSENSE NHIỄU, CÓ LỖ ============
    depth_min_m: float = 0.2
    depth_max_m: float = 1.5

    # ======================= CHUNK =======================
    chunk: int = 40                           # 40 bước = 1.6 s @ 25 Hz
    hz: float = 25.0
    val_frac: float = 0.12                    # tỉ lệ episode dùng làm val (tách theo episode, phân tầng theo tay)
    seed: int = 0                             # seed chung: khởi tạo, chia train/val, chọn tập con eval

    # ======================= MODEL =======================
    dino_name: str = "vit_large_patch14_reg4_dinov2.lvd142m"
    siglip_name: str = "vit_so400m_patch14_siglip_224"
    backbone_pretrained: bool = True          # tải trọng số timm gốc
    openvla_vision_path: Optional[str] = None # trọng số thị giác OpenVLA (tools.py extract-openvla)
    d_model: int = 512
    n_heads: int = 8
    enc_layers: int = 4
    expert_layers: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    flow_steps: int = 10

    # ======================= HUẤN LUYỆN =======================
    out_dir: str = "runs/vafm"
    batch_size: int = 48
    epochs: Optional[int] = None  # số epoch; đặt thì ghi đè total_steps = epochs * số bước/epoch
    n_flow_samples: int = 4       # số (t, noise) trên mỗi quan sát: backbone chạy 1 lần, expert 4 lần
    lr: float = 1e-4
    weight_decay: float = 0.05
    warmup_steps: int = 1000
    total_steps: int = 40000      # tổng số bước (dùng khi epochs=None)
    ema_decay: float = 0.999
    grad_clip: float = 1.0
    num_workers: int = 8

    # ============ RỦI RO 4: TAY KHÔNG HOẠT ĐỘNG ============
    inactive_weight: float = 0.1  # trọng số loss của tay đứng yên (vẫn học "giữ yên")
    gripper_weight: float = 2.0   # kẹp của tay đang làm việc quan trọng cho việc gắp
    balance_arms: bool = True     # lấy mẫu cân bằng episode tay 0 / tay 1

    # ============ RỦI RO 2 + 6: AUGMENTATION ONLINE (KHÔNG CACHE FEATURE) ============
    aug_brightness: float = 0.2
    aug_contrast: float = 0.2
    aug_saturation: float = 0.2
    aug_rgb_noise: float = 0.02
    state_noise_rad: float = 0.01   # nhiễu qpos đầu vào, chống bám qpos quá mức
    depth_noise_m: float = 0.005
    depth_hole_prob: float = 0.5

    # ======================= CHIA TRAIN / EVAL =======================
    # cache   : dùng đúng split đã ghi trong meta.json lúc chạy data.py
    # resplit : chia lại theo episode khi train với --val_frac/--seed hiện tại (không cần tạo lại cache)
    split_mode: str = "cache"
    eval_samples: int = 1000      # số mẫu val cố định dùng để eval (0 = toàn bộ tập val)
    eval_every: int = 0           # eval mỗi N bước (0 = tắt, chỉ eval theo epoch)
    eval_every_epochs: int = 1    # eval mỗi N epoch (0 = tắt)

    # ======================= LOG / CHECKPOINT =======================
    log_every: int = 50           # in + ghi log train mỗi N bước
    tensorboard: bool = True      # ghi log TensorBoard
    tb_dir: Optional[str] = None  # thư mục TensorBoard (None = <out_dir>/tb)
    save_every: int = 2000        # ghi đè last.pt mỗi N bước (0 = chỉ lưu cuối epoch)
    save_every_epochs: int = 1    # lưu epoch_XXXX.pt mỗi N epoch (0 = tắt)
    keep_ckpts: int = 3           # giữ N file epoch_XXXX.pt mới nhất (0 = giữ tất cả)
    resume: Optional[str] = None  # đường dẫn checkpoint, hoặc "auto" = <out_dir>/last.pt nếu có
    max_steps_debug: Optional[int] = None    # chạy thử vài bước

    # ======================= SUY LUẬN =======================
    exec_horizon: int = 20          # thực thi 20 bước (0.8 s) rồi dự đoán lại

    # ----------------------------------------------------------------
    @property
    def delta_cols(self):
        """Cột dùng Δq (khớp tay); kẹp dùng giá trị tuyệt đối."""
        return sorted(list(self.arm_joints[0]) + list(self.arm_joints[1]))

    def arm_cols(self, a: int):
        return list(self.arm_joints[a]) + [self.grippers[a]]

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        names = {f.name for f in fields(cls)}
        return cls(**{k: _tuplify(v) for k, v in d.items() if k in names})


# Các trường quyết định cách cache được tạo; train/inference phải dùng đúng giá trị này.
PREP_KEYS = (
    "key_rgb", "key_depth", "key_cmd", "key_state", "rgb_is_bgr", "depth_unit_m",
    "frame_stride", "action_dim", "arm_joints", "grippers", "arm_names", "crop_xyxy",
    "cache_rgb_size", "cache_depth_size", "chunk", "hz",
)


# Các trường chỉ nhận một số giá trị cố định.
CHOICES = {
    "split_mode": ("cache", "resplit"),
}


def _tuplify(v):
    if isinstance(v, list):
        return tuple(_tuplify(x) for x in v)
    return v


def _parse_value(s: str):
    try:
        return _tuplify(json.loads(s))
    except (json.JSONDecodeError, TypeError):
        return None if s in ("None", "none", "null") else s


def _str2bool(s: str) -> bool:
    if s.lower() in ("1", "true", "yes", "y", "on"):
        return True
    if s.lower() in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"cần true/false, nhận '{s}'")


def _arg_type(tp):
    """Kiểu argparse từ chú thích kiểu của trường. Tuple/Optional[Tuple] đọc dạng JSON."""
    base, optional = tp, False
    if typing.get_origin(tp) is typing.Union:
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        base, optional = args[0], len(args) < len(typing.get_args(tp))
    if base is bool:
        conv = _str2bool
    elif base in (int, float, str):
        conv = base
    else:
        return _parse_value
    if not optional:
        return conv
    return lambda s: None if s in ("None", "none", "null") else conv(s)


def _field_docs():
    """{trường: (mục, mô tả)} lấy từ tiêu đề mục và comment trong mã nguồn Config."""
    try:
        src = inspect.getsource(Config).splitlines()
    except (OSError, TypeError):
        return {}
    docs, section, pending = {}, "Khác", []
    for line in src:
        t = line.strip()
        if m := re.match(r"#\s*=+\s*(.+?)\s*=+$", t):
            section, pending = m.group(1), []
        elif t.startswith("#"):
            pending.append(t.lstrip("# ").strip())
        elif m := re.match(r"(\w+)\s*:[^=]*=.*?(?:#\s*(.*))?$", t):
            docs[m.group(1)] = (section, " ".join(pending + ([m.group(2)] if m.group(2) else [])))
            pending = []
        elif not t:
            pending = []
    return docs


def parse_config(argv=None, description="") -> Config:
    """Mặc định trong Config < file --config JSON < cờ dòng lệnh."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", type=str, default=None, help="file JSON cấu hình")
    hints = typing.get_type_hints(Config)
    docs, groups = _field_docs(), {}
    default = Config()
    for f in fields(Config):
        section, doc = docs.get(f.name, ("Khác", ""))
        g = groups.get(section) or groups.setdefault(section, p.add_argument_group(section))
        kw = dict(type=_arg_type(hints[f.name]), default=argparse.SUPPRESS, choices=CHOICES.get(f.name),
                  help=f"{doc} (mặc định: {getattr(default, f.name)})" if doc
                  else f"(mặc định: {getattr(default, f.name)})")
        if hints[f.name] is bool:            # cho phép viết tắt: --tensorboard == --tensorboard true
            kw.update(nargs="?", const=True, metavar="BOOL")
        g.add_argument(f"--{f.name}", **kw)
    args = p.parse_args(argv)
    cfg = Config()
    if args.config:
        with open(args.config) as fh:
            cfg = Config.from_dict({**cfg.to_dict(), **json.load(fh)})
    for f in fields(Config):              # SUPPRESS: cờ không truyền thì không có trong args -> giữ nguyên
        if hasattr(args, f.name):         # (nhờ vậy "--hf_repo None" vẫn ghi đè được thành None)
            setattr(cfg, f.name, getattr(args, f.name))
    return cfg
