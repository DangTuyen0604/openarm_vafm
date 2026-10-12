"""Suy luận trên robot + kiểm tra open-loop trên một episode.

Dùng trong vòng điều khiển 25 Hz (ROS2 node của bạn):

    from policy import VAPolicy, ChunkExecutor
    pol = VAPolicy("runs/vafm/best.pt")
    ex = ChunkExecutor(pol, mode="chunk", exec_horizon=20)
    ...
    q_target = ex.step(rgb, depth, qpos)   # (16,) vị trí khớp tuyệt đối -> gửi robot

    rgb  : ảnh gốc HxWx3 uint8, CÙNG thứ tự kênh với lúc ghi dữ liệu (rgb_is_bgr xử lý giống cache)
    depth: ảnh depth gốc (uint16 mm hoặc float m), cùng độ phân giải/căn chỉnh với lúc ghi
    qpos : (16,) joint_states hiện tại

Kiểm tra open-loop:
    python policy.py --ckpt runs/vafm/best.pt --h5 data/raw/episode_0.h5
"""
from __future__ import annotations

import argparse
import math
import os
import time

import h5py
import numpy as np
import torch

from config import Config
from data import (autodetect_keys, detect_active_arm, gpu_preprocess, preprocess_frame, read_block, to_meters,
                  to_rgb, valid_start)
from model import VAFlowPolicy


class VAPolicy:
    def __init__(self, ckpt_path: str, device: str | None = None, flow_steps: int | None = None, seed: int = 0):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.cfg = Config.from_dict(ck["cfg"])
        self.info = {k: ck[k] for k in ("step", "epoch", "best") if k in ck}
        if flow_steps:
            self.cfg.flow_steps = flow_steps
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = VAFlowPolicy(self.cfg, ck["stats"])
        self.model.load_trainable(ck["model"])          # trọng số EMA
        self.model.to(self.dev).eval()
        self.gen = torch.Generator(device=self.dev).manual_seed(seed)
        self.delta_cols = self.cfg.delta_cols

    @torch.no_grad()
    def predict_chunk(self, rgb: np.ndarray, depth: np.ndarray, qpos: np.ndarray) -> np.ndarray:
        """-> (K, 16) vị trí khớp tuyệt đối: tay = qpos + Δq, kẹp = giá trị tuyệt đối."""
        cfg = self.cfg
        r, d = preprocess_frame(to_rgb(rgb, cfg), to_meters(depth, cfg), cfg)
        qpos = np.asarray(qpos, np.float32)
        b = {"rgb": torch.from_numpy(r)[None].to(self.dev),
             "depth": torch.from_numpy(d.astype(np.float32))[None].to(self.dev)}
        rgb_t, dep_t = gpu_preprocess(b, cfg, train=False)
        st = torch.from_numpy(qpos)[None].to(self.dev)
        with torch.autocast(self.dev.type, dtype=torch.bfloat16, enabled=self.dev.type == "cuda"):
            ctx = self.model.encode(rgb_t, dep_t, self.model.norm_state(st))
            x = self.model.sample(ctx, generator=self.gen)
        a = self.model.unnorm_action(x)[0].cpu().numpy()
        q = a.copy()
        q[:, self.delta_cols] += qpos[self.delta_cols]
        return q


class ChunkExecutor:
    """mode='chunk'   : thực thi exec_horizon bước rồi dự đoán lại (rẻ, mặc định).
       mode='ensemble': dự đoán mỗi bước + temporal ensembling kiểu ACT (cần GPU đủ nhanh cho 25 Hz)."""

    def __init__(self, policy: VAPolicy, mode: str = "chunk", exec_horizon: int | None = None, ens_k: float = 0.01):
        self.p, self.mode, self.k = policy, mode, ens_k
        self.h = exec_horizon or policy.cfg.exec_horizon
        self.reset()

    def reset(self):
        self.plan, self.i, self.t, self.history = None, 0, 0, []

    def step(self, rgb, depth, qpos) -> np.ndarray:
        K = self.p.cfg.chunk
        if self.mode == "chunk":
            if self.plan is None or self.i >= min(self.h, K):
                self.plan, self.i = self.p.predict_chunk(rgb, depth, qpos), 0
            out = self.plan[self.i]
            self.i += 1
            return out
        self.history.append((self.t, self.p.predict_chunk(rgb, depth, qpos)))
        self.history = [(s, c) for s, c in self.history if self.t - s < K]
        preds = np.stack([c[self.t - s] for s, c in self.history])     # cũ nhất trước
        w = np.exp(-self.k * np.arange(len(preds)))
        self.t += 1
        return (preds * w[:, None]).sum(0) / w.sum()


def open_loop_eval(ckpt, h5_path, stride=None, device=None) -> dict:
    """`ckpt`: đường dẫn checkpoint hoặc VAPolicy đã nạp (tránh nạp lại backbone cho mỗi episode)."""
    pol = ckpt if isinstance(ckpt, VAPolicy) else VAPolicy(ckpt, device)
    cfg = pol.cfg
    stride = stride or cfg.exec_horizon
    with h5py.File(h5_path, "r") as f:
        keys = autodetect_keys(f, cfg)
        cmd = np.asarray(f[keys["cmd"]][:], np.float32)
        st = np.asarray(f[keys["state"]][:], np.float32)
        s0 = valid_start(cmd, st, h5_path)
        cmd, st = cmd[s0::cfg.frame_stride], st[s0::cfg.frame_stride]
        arm, _, _ = detect_active_arm(cmd, cfg)
        T, K = len(cmd), cfg.chunk
        joints = list(cfg.arm_joints[arm])
        g = cfg.grippers[arm]
        errs, holds, gerrs, dts = [], [], [], []
        for t in range(0, T, stride):
            raw_i = np.array([s0 + t * cfg.frame_stride])
            rgb = read_block(f[keys["rgb"]], raw_i, "rgb")[0]
            dep = read_block(f[keys["depth"]], raw_i, "depth")[0]
            t0 = time.time()
            q = pol.predict_chunk(rgb, dep, st[t])
            dts.append(time.time() - t0)
            n = min(K, T - t)
            gt = cmd[t:t + n]
            errs.append(np.abs(q[:n, joints] - gt[:, joints]).mean())
            holds.append(np.abs(st[t, joints] - gt[:, joints]).mean())
            gerrs.append(np.abs(q[:n, g] - gt[:, g]).mean())
    deg = 180 / math.pi
    print(f"Episode {h5_path} | tay {cfg.arm_names[arm]} | {len(errs)} lần dự đoán")
    print(f"  MAE khớp tay   : {np.mean(errs) * deg:.2f}°  (baseline đứng yên {np.mean(holds) * deg:.2f}°)")
    print(f"  MAE kẹp        : {np.mean(gerrs):.4f}")
    print(f"  Thời gian/chunk: {np.median(dts[1:] or dts) * 1000:.0f} ms (median)")
    return {"episode": os.path.basename(h5_path), "arm": cfg.arm_names[arm], "n_pred": len(errs),
            "arm_mae_deg": float(np.mean(errs) * deg), "hold_mae_deg": float(np.mean(holds) * deg),
            "grip_mae": float(np.mean(gerrs)), "ms_per_chunk": float(np.median(dts[1:] or dts) * 1000)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--h5", required=True)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    open_loop_eval(a.ckpt, a.h5, a.stride, a.device)
