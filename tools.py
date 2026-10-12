"""Công cụ kiểm tra trước khi train.

  python tools.py inspect data/raw/episode_0.h5          # in cấu trúc HDF5 + key tự dò
  python tools.py preview-crop --crop_xyxy "[104,48,536,480]"   # RỦI RO 1: chỉnh vùng crop
  python tools.py joints                                  # kiểm tra bố cục 16 khớp / tay hoạt động
  python tools.py extract-openvla --out weights/openvla_vision.safetensors
"""
from __future__ import annotations

import os
import sys

import cv2
import h5py
import numpy as np
import torch

from config import parse_config
from data import (autodetect_keys, crop_box, detect_active_arm, find_h5, gpu_preprocess,
                  preprocess_frame, print_h5_tree, read_block, to_meters, to_rgb, valid_start)


def cmd_inspect(argv):
    path = argv[0]
    cfg = parse_config(argv[1:])
    with h5py.File(path, "r") as f:
        print_h5_tree(f)
        print("Key tự dò:", autodetect_keys(f, cfg))


def cmd_preview_crop(argv):
    """Lưu crop_preview.png: hàng trên = ảnh gốc + khung crop, hàng dưới = ảnh model thật sự thấy
    (RGB 224 và depth 128 sau crop giữa lúc eval). Vật + kẹp phải đủ lớn ở hàng dưới."""
    cfg = parse_config(argv)
    files = find_h5(cfg.raw_dir)[:: max(1, len(find_h5(cfg.raw_dir)) // 6)][:6]
    tops, bots = [], []
    for fp in files:
        with h5py.File(fp, "r") as f:
            k = autodetect_keys(f, cfg)
            T = f[k["cmd"]].shape[0]
            i = np.array([T // 2])
            rgb = to_rgb(read_block(f[k["rgb"]], i, "rgb")[0], cfg)
            dep = to_meters(read_block(f[k["depth"]], i, "depth")[0], cfg)
        H, W = rgb.shape[:2]
        x0, y0, x1, y1 = crop_box(cfg, W, H, W, H)
        top = rgb.copy()
        cv2.rectangle(top, (x0, y0), (x1 - 1, y1 - 1), (255, 0, 0), 3)
        tops.append(cv2.resize(top, (320, int(320 * H / W))))
        r, d = preprocess_frame(rgb, dep, cfg)
        rt, dt = gpu_preprocess({"rgb": torch.from_numpy(r)[None],
                                 "depth": torch.from_numpy(d.astype(np.float32))[None]}, cfg, train=False)
        r224 = (rt[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        dvis = cv2.applyColorMap((dt[0, 0].numpy() * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
        dvis[dt[0, 1].numpy() < 0.5] = 0
        dvis = cv2.resize(dvis, (96, 96), interpolation=cv2.INTER_NEAREST)
        bot = np.zeros((224, 320, 3), np.uint8)
        bot[:, :224] = r224
        bot[:96, 224:] = dvis
        bots.append(bot)
    h = max(t.shape[0] for t in tops)
    tops = [np.pad(t, ((0, h - t.shape[0]), (0, 0), (0, 0))) for t in tops]
    grid = np.concatenate([np.concatenate(tops, 1), np.concatenate(bots, 1)], 0)
    cv2.imwrite("crop_preview.png", grid[..., ::-1])
    vat = (x1 - x0, y1 - y0)
    print(f"Đã lưu crop_preview.png | crop {vat[0]}x{vat[1]} px -> {cfg.img_size}px "
          f"(thu nhỏ {max(vat) * cfg.crop_scale_eval / cfg.img_size:.2f}x; cả khung {max(W, H) / cfg.img_size:.2f}x)")
    print("Nếu màu bị đảo xanh/đỏ ở hàng dưới: thêm --rgb_is_bgr true")


def cmd_joints(argv):
    cfg = parse_config(argv)
    np.set_printoptions(precision=2, suppress=True, linewidth=200)
    print("Biên độ joint_commands mỗi khớp (rad) theo episode. Tay hoạt động phải có các cột lớn rõ rệt.")
    print(f"Tay 0 ({cfg.arm_names[0]}): khớp {cfg.arm_joints[0]} kẹp {cfg.grippers[0]} | "
          f"Tay 1 ({cfg.arm_names[1]}): khớp {cfg.arm_joints[1]} kẹp {cfg.grippers[1]}")
    for fp in find_h5(cfg.raw_dir):
        with h5py.File(fp, "r") as f:
            k = autodetect_keys(f, cfg)
            cmd = np.asarray(f[k["cmd"]][:], np.float32)
            st = np.asarray(f[k["state"]][:], np.float32)
        s0 = valid_start(cmd, st, fp)
        cmd, st = cmd[s0:], st[s0:]
        arm, motion, ratio = detect_active_arm(cmd, cfg)
        lag = np.abs(cmd - st).mean(0)
        print(f"{os.path.basename(fp)[:28]:28s} {cfg.arm_names[arm]:5s} tỉ lệ {ratio:5.1f} "
              f"| biên độ {cmd.max(0) - cmd.min(0)} | |cmd-state| TB {lag.mean():.3f}")


def cmd_extract_openvla(argv):
    """Chỉ tải các shard chứa vision_backbone.* của openvla/openvla-7b và tách ra một file nhỏ."""
    import json
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from safetensors.torch import save_file
    out = argv[argv.index("--out") + 1] if "--out" in argv else "weights/openvla_vision.safetensors"
    repo = argv[argv.index("--repo") + 1] if "--repo" in argv else "openvla/openvla-7b"
    idx = json.load(open(hf_hub_download(repo, "model.safetensors.index.json")))["weight_map"]
    keys = [k for k in idx if k.startswith("vision_backbone.")]
    shards = sorted({idx[k] for k in keys})
    print(f"{len(keys)} tensor thị giác trong {shards}")
    sd = {}
    for sh in shards:
        p = hf_hub_download(repo, sh)
        with safe_open(p, "pt") as f:
            for k in keys:
                if idx[k] == sh:
                    sd[k[len("vision_backbone."):]] = f.get_tensor(k).contiguous()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    save_file(sd, out)
    print(f"Đã lưu {out}. Dùng: --openvla_vision_path {out}")


if __name__ == "__main__":
    cmds = {"inspect": cmd_inspect, "preview-crop": cmd_preview_crop,
            "joints": cmd_joints, "extract-openvla": cmd_extract_openvla}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        sys.exit(__doc__)
    cmds[sys.argv[1]](sys.argv[2:])
