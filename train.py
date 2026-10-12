"""Huấn luyện VA flow-matching policy.

    python train.py --cache_dir data/cache --out_dir runs/vafm
    python train.py --cache_dir data/cache --openvla_vision_path weights/openvla_vision.safetensors
    python train.py --epochs 30 --batch_size 32 --lr 2e-4 --save_every_epochs 5 --keep_ckpts 3
    python train.py --split_mode resplit --val_frac 0.2 --eval_samples 2000   # chia lại train/val
    python train.py --out_dir runs/vafm --resume auto                         # chạy tiếp từ last.pt
    python train.py --help                                                    # xem toàn bộ cờ

    tensorboard --logdir runs        # loss/train vs loss/val, mse_*, val/*, opt/*, perf/*

File trong out_dir:
    last.pt            : ghi đè mỗi --save_every bước và cuối mỗi epoch (dùng để --resume)
    epoch_XXXX.pt      : mỗi --save_every_epochs epoch, giữ --keep_ckpts file mới nhất
    best.pt            : điểm eval (score) tốt nhất
    config.json, split.json (episode train/val), log.jsonl, tb/ (TensorBoard)

Log theo dõi (RỦI RO 4: không nhìn loss tổng):
    mse_arm    : loss flow của 7 khớp tay ĐANG làm việc
    mse_grip   : loss flow của kẹp tay đang làm việc
    mse_idle   : loss flow của tay đứng yên (dễ, không nói lên gì)
Eval (đơn vị vật lý, trọng số EMA, lấy mẫu 10 bước Euler):
    arm_mae_deg@k : sai số tuyệt đối Δq tay làm việc (độ) tại bước k của chunk
    hold_mae_deg  : baseline "đứng yên" (Δq = 0). Model phải thấp hơn rõ rệt con số này.
    grip_mae      : sai số kẹp; grip_hold = baseline giữ nguyên kẹp
    idle_drift_deg: tay đứng yên bị model cho trôi bao nhiêu
"""
from __future__ import annotations

import glob
import json
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from config import parse_config
from data import ChunkDataset, gpu_preprocess, load_prep_cfg, split_episodes
from model import VAFlowPolicy, sample_flow_t


def to_dev(batch, dev):
    return {k: v.to(dev, non_blocking=True) for k, v in batch.items()}


def masks_for(cfg, arm, pad, dev):
    """Mask cột (B,D): khớp tay làm việc / kẹp tay làm việc / tay đứng yên; valid (B,K)."""
    D = cfg.action_dim
    act = torch.zeros(2, D, dtype=torch.bool, device=dev)
    grip = torch.zeros(2, D, dtype=torch.bool, device=dev)
    for a in (0, 1):
        act[a, cfg.arm_cols(a)] = True
        grip[a, cfg.grippers[a]] = True
    act, grip = act[arm], grip[arm]
    return act & ~grip, grip, ~act, ~pad


def loss_weights(cfg, m_arm, m_grip, m_idle, valid):
    w = m_arm.float() + cfg.gripper_weight * m_grip.float() + cfg.inactive_weight * m_idle.float()
    return w[:, None, :] * valid[:, :, None].float()


def masked_mean(x, m):
    m = m.float()
    return (x * m).sum() / m.sum().clamp_min(1)


def flow_loss(model, ctx, a, w, generator=None):
    """Loss flow-matching có trọng số. Trả về (loss vô hướng, sai số bình phương (B,K,D))."""
    t = sample_flow_t(a.shape[0], a.device, generator)
    noise = torch.randn(a.shape, device=a.device, generator=generator)
    x_t = t[:, None, None] * noise + (1 - t[:, None, None]) * a
    v = model.velocity(x_t, t, ctx)
    err = (v.float() - (noise - a)) ** 2
    return (err * w).sum() / w.sum(), err


def flow_parts(err, m_arm, m_grip, m_idle, valid):
    V = valid[:, :, None]
    return {"mse_arm": masked_mean(err, V & m_arm[:, None]),
            "mse_grip": masked_mean(err, V & m_grip[:, None]),
            "mse_idle": masked_mean(err, V & m_idle[:, None])}


@torch.no_grad()
def evaluate(model, loader, cfg, dev):
    """Chạy trên toàn bộ `loader` (tập con val cố định) -> kết quả so sánh được giữa các lần eval."""
    model.eval()
    K = cfg.chunk
    ks = sorted({0, K // 2, K - 1})
    acc = {k: [] for k in ("loss", "mse_arm", "mse_grip", "mse_idle")}
    acc.update({f"arm_mae_deg@{k}": [] for k in ks})
    acc.update({"arm_mae_deg": [], "hold_mae_deg": [], "grip_mae": [], "grip_hold": [], "idle_drift_deg": []})
    g = torch.Generator(device=dev).manual_seed(0)
    deg = 180.0 / math.pi
    for batch in loader:
        b = to_dev(batch, dev)
        rgb, depth = gpu_preprocess(b, cfg, train=False)
        m_arm, m_grip, m_idle, valid = masks_for(cfg, b["arm"], b["pad"], dev)
        w = loss_weights(cfg, m_arm, m_grip, m_idle, valid)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            ctx = model.encode(rgb, depth, model.norm_state(b["state"]))
            loss, err_f = flow_loss(model, ctx, model.norm_action(b["action"]), w, g)
            x = model.sample(ctx, generator=g)
        acc["loss"].append(loss.item())
        for k, v in flow_parts(err_f, m_arm, m_grip, m_idle, valid).items():
            acc[k].append(v.item())
        pred, gt = model.unnorm_action(x), b["action"]
        err = (pred - gt).abs()
        V = valid[:, :, None]
        acc["arm_mae_deg"].append(masked_mean(err, V & m_arm[:, None]).item() * deg)
        acc["hold_mae_deg"].append(masked_mean(gt.abs(), V & m_arm[:, None]).item() * deg)
        for k in ks:
            acc[f"arm_mae_deg@{k}"].append(masked_mean(err[:, k], valid[:, k:k + 1] & m_arm).item() * deg)
        acc["grip_mae"].append(masked_mean(err, V & m_grip[:, None]).item())
        hold = (gt - b["state"][:, None]).abs()
        acc["grip_hold"].append(masked_mean(hold, V & m_grip[:, None]).item())
        idle_j = m_idle.clone()
        idle_j[:, list(cfg.grippers)] = False
        acc["idle_drift_deg"].append(masked_mean((pred - gt).abs(), V & idle_j[:, None]).item() * deg)
    model.train()
    ev = {k: float(np.mean(v)) for k, v in acc.items() if v}
    # so với baseline đứng yên -> không phụ thuộc đơn vị của kẹp
    ev["score"] = (ev["arm_mae_deg"] / max(ev["hold_mae_deg"], 1e-6)
                   + ev["grip_mae"] / max(ev["grip_hold"], 1e-6))
    return ev


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].lerp_(p.detach(), 1 - self.decay)

    @torch.no_grad()
    def swap(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow:
                tmp = p.detach().clone()
                p.copy_(self.shadow[n])
                self.shadow[n].copy_(tmp)


def save_ckpt(path, model, ema, opt, sched, step, epoch, cfg, stats, best):
    ema.swap(model)
    ema_sd = {k: v.cpu() for k, v in model.trainable_state_dict().items()}
    ema.swap(model)
    tmp = path + ".tmp"                                      # ghi xong mới thay -> không hỏng file khi bị ngắt
    torch.save({
        "model": ema_sd,                                     # EMA -> dùng cho suy luận
        "raw": {k: v.cpu() for k, v in model.trainable_state_dict().items()},
        "ema_shadow": {k: v.cpu() for k, v in ema.shadow.items()},
        "opt": opt.state_dict(), "sched": sched.state_dict(),
        "step": step, "epoch": epoch, "best": best, "cfg": cfg.to_dict(),
        "stats": {k: np.asarray(v) for k, v in stats.items()},
    }, tmp)
    os.replace(tmp, path)


def rotate_ckpts(out_dir, keep):
    """Chỉ giữ `keep` file epoch_XXXX.pt mới nhất (keep <= 0: giữ tất cả)."""
    if keep > 0:
        for p in sorted(glob.glob(os.path.join(out_dir, "epoch_*.pt")))[:-keep]:
            os.remove(p)


def make_writer(cfg, step):
    if not cfg.tensorboard:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("[train] Chưa cài tensorboard (pip install tensorboard) -> tắt log TensorBoard")
        return None
    tb_dir = cfg.tb_dir or os.path.join(cfg.out_dir, "tb")
    # purge_step: khi resume từ checkpoint cũ, bỏ các điểm log sau bước đó để đồ thị không chồng lên nhau
    w = SummaryWriter(tb_dir, purge_step=step + 1 if step > 0 else None)
    w.add_text("config", "```json\n" + json.dumps(cfg.to_dict(), indent=1) + "\n```", step)
    print(f"[train] TensorBoard: tensorboard --logdir {tb_dir}")
    return w


def tb_log(writer, scalars, step):
    if writer is not None:
        for k, v in scalars.items():
            writer.add_scalar(k, v, step)


def main(cfg=None):
    cfg = cfg or parse_config(description="Train VA flow-matching policy")
    cfg, meta, stats = load_prep_cfg(cfg, cfg.cache_dir)
    os.makedirs(cfg.out_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ---- chia train / val theo episode
    if cfg.split_mode == "resplit":
        split_episodes(meta["episodes"], cfg.val_frac, cfg.seed)
        print(f"[train] Chia lại train/val: val_frac={cfg.val_frac} seed={cfg.seed} "
              "(stats chuẩn hoá vẫn lấy từ split lúc tạo cache)")
    split = {s_: [e["file"] for e in meta["episodes"] if e["split"] == s_] for s_ in ("train", "val")}
    with open(os.path.join(cfg.out_dir, "split.json"), "w") as fh:
        json.dump({"mode": cfg.split_mode, "val_frac": cfg.val_frac, "seed": cfg.seed, **split}, fh, indent=1)

    train_ds = ChunkDataset(cfg, meta, "train")
    val_ds = ChunkDataset(cfg, meta, "val")
    sampler = (WeightedRandomSampler(train_ds.sample_weights(), len(train_ds), replacement=True)
               if cfg.balance_arms else None)
    kw = dict(num_workers=cfg.num_workers, pin_memory=dev.type == "cuda",
              persistent_workers=cfg.num_workers > 0)
    train_dl = DataLoader(train_ds, cfg.batch_size, sampler=sampler, shuffle=sampler is None,
                          drop_last=True, **kw)
    # tập con val CỐ ĐỊNH (chọn 1 lần theo seed) -> các lần eval so sánh được với nhau
    val_dl = None
    if len(val_ds):
        n_eval = len(val_ds) if cfg.eval_samples <= 0 else min(cfg.eval_samples, len(val_ds))
        idx = np.sort(np.random.default_rng(cfg.seed).choice(len(val_ds), n_eval, replace=False))
        val_dl = DataLoader(Subset(val_ds, idx.tolist()), cfg.batch_size, shuffle=False, drop_last=False, **kw)
    spe = len(train_dl)                                    # số bước / epoch
    if spe == 0:
        raise SystemExit(f"[train] Tập train ({len(train_ds)} mẫu) nhỏ hơn batch_size {cfg.batch_size}")
    if cfg.epochs:
        cfg.total_steps = cfg.epochs * spe
    n_epochs = math.ceil(cfg.total_steps / spe)

    model = VAFlowPolicy(cfg, stats).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    decay = [p for p in params if p.ndim >= 2]
    no_decay = [p for p in params if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=cfg.lr, betas=(0.9, 0.95))

    def lr_lambda(s):
        if s < cfg.warmup_steps:
            return (s + 1) / cfg.warmup_steps
        p = (s - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    ema = EMA(model, cfg.ema_decay)
    step, best = 0, float("inf")
    if cfg.resume == "auto":
        last = os.path.join(cfg.out_dir, "last.pt")
        cfg.resume = last if os.path.exists(last) else None
    if cfg.resume:
        ck = torch.load(cfg.resume, map_location="cpu", weights_only=False)
        model.load_trainable(ck["raw"])
        ema.shadow = {k: v.to(dev) for k, v in ck["ema_shadow"].items()}
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        step, best = ck["step"], ck.get("best", best)
        print(f"[train] Tiếp tục từ {cfg.resume} (bước {step}, epoch {step / spe:.2f})")

    n_tr = sum(p.numel() for p in params)
    n_all = sum(p.numel() for p in model.parameters())
    n_val_eps = len(split["val"])
    print(f"[train] {len(train_ds)} mẫu train ({len(split['train'])} ep) / {len(val_ds)} val ({n_val_eps} ep), "
          f"eval trên {len(val_dl.dataset) if val_dl else 0} mẫu | token ngữ cảnh {model.n_ctx} "
          f"| tham số huấn luyện {n_tr / 1e6:.1f}M / tổng {n_all / 1e6:.0f}M | {dev}")
    print(f"[train] {spe} bước/epoch x {n_epochs} epoch = {cfg.total_steps} bước")
    with open(os.path.join(cfg.out_dir, "config.json"), "w") as fh:
        json.dump(cfg.to_dict(), fh, indent=1)
    log_f = open(os.path.join(cfg.out_dir, "log.jsonl"), "a")
    writer = make_writer(cfg, step)

    def run_eval():
        nonlocal best
        ema.swap(model)
        ev = evaluate(model, val_dl, cfg, dev)
        ema.swap(model)
        print(f"[eval {step:6d}] " + " | ".join(f"{k} {v:.3f}" for k, v in ev.items()))
        log_f.write(json.dumps({"step": step, "epoch": step / spe, **{f"val/{k}": v for k, v in ev.items()}}) + "\n")
        log_f.flush()
        flow = ("loss", "mse_arm", "mse_grip", "mse_idle")
        tb_log(writer, {f"{k}/val": ev[k] for k in flow}, step)
        tb_log(writer, {f"val/{k.replace('@', '_step')}": v for k, v in ev.items() if k not in flow}, step)
        if ev["score"] < best:
            best = ev["score"]
            save_ckpt(os.path.join(cfg.out_dir, "best.pt"),
                      model, ema, opt, sched, step, step / spe, cfg, stats, best)
            print(f"[eval] best mới ({best:.3f}) -> best.pt")
        tb_log(writer, {"val/best_score": best}, step)
        return step

    def save_last():
        save_ckpt(os.path.join(cfg.out_dir, "last.pt"), model, ema, opt, sched, step, step / spe, cfg, stats, best)

    model.train()
    Fs = cfg.n_flow_samples
    run, ep_run, n_ep, t0 = {}, {}, 0, time.time()
    last_eval = -1
    last_step = cfg.total_steps if cfg.max_steps_debug is None else min(cfg.total_steps, step + cfg.max_steps_debug)
    while step < last_step:
        for batch in train_dl:
            b = to_dev(batch, dev)
            with torch.no_grad():
                rgb, depth = gpu_preprocess(b, cfg, train=True)
                state = b["state"]
                if cfg.state_noise_rad > 0:      # nhiễu chỉ ở đầu vào; nhãn Δq tính theo qpos thật
                    state = state + torch.randn_like(state) * cfg.state_noise_rad
                a = model.norm_action(b["action"])
                # backbone chạy 1 lần, expert học Fs mẫu (t, nhiễu)
                m_arm, m_grip, m_idle, valid = (m.repeat_interleave(Fs, 0)
                                                for m in masks_for(cfg, b["arm"], b["pad"], dev))
                w = loss_weights(cfg, m_arm, m_grip, m_idle, valid)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                ctx = model.encode(rgb, depth, model.norm_state(state)).repeat_interleave(Fs, 0)
                loss, err = flow_loss(model, ctx, a.repeat_interleave(Fs, 0), w)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step(); sched.step(); ema.update(model)
            step += 1

            with torch.no_grad():
                vals = {"loss": loss, **flow_parts(err, m_arm, m_grip, m_idle, valid), "gnorm": gn}
                for k, val in vals.items():
                    run[k] = run.get(k, 0.0) + float(val)
                    ep_run[k] = ep_run.get(k, 0.0) + float(val)
                n_ep += 1
            if step % cfg.log_every == 0 or step == last_step:
                n = cfg.log_every if step % cfg.log_every == 0 else (step % cfg.log_every)
                dt = time.time() - t0
                rec = {k: v / n for k, v in run.items()}
                rec.update(step=step, epoch=step / spe, lr=sched.get_last_lr()[0], s_per_it=dt / n)
                if dev.type == "cuda":
                    rec["gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
                    torch.cuda.reset_peak_memory_stats()
                print(f"[{step:6d} ep {rec['epoch']:6.2f}] loss {rec['loss']:.4f} | arm {rec['mse_arm']:.4f} "
                      f"grip {rec['mse_grip']:.4f} idle {rec['mse_idle']:.4f} | gn {rec['gnorm']:.2f} "
                      f"lr {rec['lr']:.2e} {rec['s_per_it']:.2f}s/it"
                      + (f" | VRAM đỉnh {rec['gpu_mem_gb']:.1f} GB" if "gpu_mem_gb" in rec else ""))
                log_f.write(json.dumps(rec) + "\n"); log_f.flush()
                tb = {f"{k}/train": rec[k] for k in ("loss", "mse_arm", "mse_grip", "mse_idle")}
                tb.update({"opt/grad_norm": rec["gnorm"], "opt/lr": rec["lr"], "progress/epoch": rec["epoch"],
                           "perf/s_per_it": rec["s_per_it"], "perf/samples_per_s": n * cfg.batch_size / dt})
                if "gpu_mem_gb" in rec:
                    tb["perf/gpu_mem_gb"] = rec["gpu_mem_gb"]
                tb_log(writer, tb, step)
                run, t0 = {}, time.time()

            if val_dl is not None and cfg.eval_every > 0 and step % cfg.eval_every == 0:
                last_eval = run_eval()
            if cfg.save_every > 0 and step % cfg.save_every == 0:
                save_last()
            if step % spe == 0 or step >= last_step:
                break       # hết epoch (kể cả khi resume giữa epoch) hoặc hết số bước

        if step % spe == 0:                                # ---- cuối epoch
            epoch = step // spe
            ep = {k: v / max(n_ep, 1) for k, v in ep_run.items()}
            print(f"[epoch {epoch}/{n_epochs}] train loss {ep.get('loss', float('nan')):.4f}")
            tb_log(writer, {f"{k}/train_epoch": v for k, v in ep.items() if k != "gnorm"}, step)
            ep_run, n_ep = {}, 0
            if val_dl is not None and cfg.eval_every_epochs > 0 and epoch % cfg.eval_every_epochs == 0 \
                    and last_eval != step:
                last_eval = run_eval()
            save_last()
            if cfg.save_every_epochs > 0 and epoch % cfg.save_every_epochs == 0:
                save_ckpt(os.path.join(cfg.out_dir, f"epoch_{epoch:04d}.pt"),
                          model, ema, opt, sched, step, epoch, cfg, stats, best)
                rotate_ckpts(cfg.out_dir, cfg.keep_ckpts)
                print(f"[epoch {epoch}] -> epoch_{epoch:04d}.pt")

    if val_dl is not None and last_eval != step:          # eval lần cuối nếu chưa eval ở bước này
        run_eval()
    save_last()
    log_f.close()
    if writer is not None:
        writer.close()
    print(f"[train] Xong. best score {best:.3f}")


if __name__ == "__main__":
    main()
