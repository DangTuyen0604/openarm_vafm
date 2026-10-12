"""Một lệnh cho cả quy trình: tạo cache (nếu chưa có) -> train -> eval cuối -> ghi kết quả ra file.

    nohup python pipeline.py <các cờ giống train.py / data.py> > /dev/null 2>&1 &

Sau đó chỉ cần xem trong <out_dir>:
    STATUS        : RUNNING (đang ở bước nào) / DONE / FAILED / STOPPED
    results.md    : bảng kết quả dễ đọc (chỉ có khi DONE)
    results.json  : cùng nội dung, dạng máy đọc
    pipeline.log  : toàn bộ những gì được in ra (kể cả traceback khi lỗi)

Eval cuối dùng best.pt (trọng số EMA; nếu chưa có thì last.pt):
    1. TOÀN BỘ tập val (không lấy tập con), chia thêm theo từng tay
    2. open-loop trên từng episode val đọc thẳng từ file HDF5 gốc (cần --raw_dir còn dữ liệu)

Cờ riêng của pipeline (ngoài các cờ của train.py):
    --skip_train          chỉ eval lại một lần train đã xong trong out_dir
    --open_loop_max N     số episode val chạy open-loop (mặc định 0 = tất cả, -1 = bỏ qua)

Chạy lại đúng lệnh cũ (có --resume auto) sau khi bị ngắt: train tiếp từ last.pt rồi eval.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import traceback

import numpy as np
from torch.utils.data import DataLoader, Subset

import train
from config import parse_config
from data import ChunkDataset, find_h5, load_prep_cfg, prepare
from policy import VAPolicy, open_loop_eval


class Tee:
    """Ghi đồng thời ra màn hình và pipeline.log."""

    def __init__(self, stream, fh):
        self.stream, self.fh = stream, fh

    def write(self, s):
        self.stream.write(s)
        self.fh.write(s)
        self.fh.flush()

    def flush(self):
        self.stream.flush()
        self.fh.flush()

    def isatty(self):
        return False


def set_status(out_dir, text):
    with open(os.path.join(out_dir, "STATUS"), "w") as fh:
        fh.write(f"{text}\n{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    print(f"[pipeline] {text}")


def final_eval(ckpt_path, out_dir, open_loop_max):
    pol = VAPolicy(ckpt_path)
    cfg, model, dev = pol.cfg, pol.model, pol.dev
    _, meta, _ = load_prep_cfg(cfg, cfg.cache_dir)
    with open(os.path.join(out_dir, "split.json")) as fh:
        split = json.load(fh)
    val_files = set(split["val"])
    for e in meta["episodes"]:                    # đúng split đã dùng khi train
        e["split"] = "val" if e["file"] in val_files else "train"
    val_ds = ChunkDataset(cfg, meta, "val")
    if not len(val_ds):
        raise SystemExit("[pipeline] Tập val rỗng (val_frac = 0?) -> không có gì để eval")

    def run(ds):
        dl = DataLoader(ds, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                        pin_memory=dev.type == "cuda")
        ev = train.evaluate(model, dl, cfg, dev)
        model.eval()                              # evaluate() trả model về train()
        return ev

    print(f"[pipeline] Eval {ckpt_path} trên toàn bộ val: {len(val_ds)} mẫu / {len(val_files)} episode")
    res = {"ckpt": ckpt_path, **pol.info, "flow_steps": cfg.flow_steps, "n_val_samples": len(val_ds), "n_val_episodes": len(val_files),
           "val": run(val_ds), "val_by_arm": {}}
    arms = val_ds.index[:, 2]
    for a in (0, 1):
        idx = np.where(arms == a)[0]
        if len(idx):
            res["val_by_arm"][cfg.arm_names[a]] = {"n": int(len(idx)), **run(Subset(val_ds, idx.tolist()))}

    res["open_loop"] = []
    if open_loop_max >= 0:
        raw = {os.path.basename(p): p for p in find_h5(cfg.raw_dir)}
        eps = sorted(val_files)[:open_loop_max or None]
        missing = [f for f in eps if f not in raw]
        if missing:
            print(f"[pipeline] Bỏ qua open-loop {len(missing)} episode không có trong {cfg.raw_dir}")
        for f in eps:
            if f in raw:
                res["open_loop"].append(open_loop_eval(pol, raw[f]))
    return res


def eval_history(out_dir):
    """Các lần eval trong lúc train (từ log.jsonl) -> xem có overfit không."""
    rows = []
    path = os.path.join(out_dir, "log.jsonl")
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                r = json.loads(line)
                if "val/score" in r:
                    rows.append({"step": r["step"], "epoch": r.get("epoch"), "loss": r.get("val/loss"),
                                 "arm_mae_deg": r.get("val/arm_mae_deg"), "score": r["val/score"]})
    return rows


def write_report(out_dir, res, hist, t_start):
    res["hours"] = (time.time() - t_start) / 3600
    res["eval_history"] = hist
    with open(os.path.join(out_dir, "results.json"), "w") as fh:
        json.dump(res, fh, indent=1)

    v = res["val"]
    ratio = lambda a, b: a / max(b, 1e-9)
    L = [f"# Kết quả: {out_dir}", "",
         f"- Checkpoint: `{res['ckpt']}` (bước {res.get('step')}, epoch {res.get('epoch', 0):.1f}, "
         f"score lúc train {res.get('best', float('nan')):.3f})",
         f"- Eval trên TOÀN BỘ val: {res['n_val_samples']} mẫu, {res['n_val_episodes']} episode, "
         f"trọng số EMA, Euler {res['flow_steps']} bước",
         f"- Tổng thời gian pipeline lần chạy này: {res['hours']:.2f} giờ", "",
         "## Tóm tắt", "",
         f"- Khớp tay: model sai **{v['arm_mae_deg']:.2f}°**, baseline đứng yên {v['hold_mae_deg']:.2f}° "
         f"-> sai bằng **{ratio(v['arm_mae_deg'], v['hold_mae_deg']):.0%}** baseline (càng thấp càng tốt)",
         f"- Kẹp: model sai **{v['grip_mae']:.4f}**, baseline giữ nguyên {v['grip_hold']:.4f} "
         f"-> **{ratio(v['grip_mae'], v['grip_hold']):.0%}** baseline",
         f"- Tay đứng yên bị trôi: {v['idle_drift_deg']:.2f}°",
         f"- Val flow loss: {v['loss']:.4f} | score: {v['score']:.3f}", "",
         "## Theo từng tay", "",
         "| Tay | Mẫu | arm MAE (°) | hold (°) | % baseline | grip MAE | grip hold | idle drift (°) |",
         "|---|---|---|---|---|---|---|---|"]
    for name, a in res["val_by_arm"].items():
        L.append(f"| {name} | {a['n']} | {a['arm_mae_deg']:.2f} | {a['hold_mae_deg']:.2f} | "
                 f"{ratio(a['arm_mae_deg'], a['hold_mae_deg']):.0%} | {a['grip_mae']:.4f} | "
                 f"{a['grip_hold']:.4f} | {a['idle_drift_deg']:.2f} |")
    ks = sorted(int(k.split("@")[1]) for k in v if k.startswith("arm_mae_deg@"))
    L += ["", "## Sai số khớp tay theo bước trong chunk", "",
          "| " + " | ".join(f"bước {k}" for k in ks) + " |", "|" + "---|" * len(ks),
          "| " + " | ".join(f"{v[f'arm_mae_deg@{k}']:.2f}°" for k in ks) + " |"]
    if res["open_loop"]:
        ol = res["open_loop"]
        L += ["", "## Open-loop trên episode val (file HDF5 gốc)", "",
              "| Episode | Tay | Lần dự đoán | arm MAE (°) | hold (°) | grip MAE | ms/chunk |",
              "|---|---|---|---|---|---|---|"]
        for r in ol:
            L.append(f"| {r['episode']} | {r['arm']} | {r['n_pred']} | {r['arm_mae_deg']:.2f} | "
                     f"{r['hold_mae_deg']:.2f} | {r['grip_mae']:.4f} | {r['ms_per_chunk']:.0f} |")
        m = {k: float(np.mean([r[k] for r in ol])) for k in ("arm_mae_deg", "hold_mae_deg", "grip_mae", "ms_per_chunk")}
        L.append(f"| **trung bình** | | | **{m['arm_mae_deg']:.2f}** | {m['hold_mae_deg']:.2f} | "
                 f"{m['grip_mae']:.4f} | {m['ms_per_chunk']:.0f} |")
    if hist:
        L += ["", "## Lịch sử eval trong lúc train (tập con val)", "",
              "| Bước | Epoch | val loss | arm MAE (°) | score |", "|---|---|---|---|---|"]
        best = min(h["score"] for h in hist)
        for h in hist:
            mark = " ← best" if h["score"] == best else ""
            L.append(f"| {h['step']} | {h['epoch'] or 0:.1f} | {h['loss']:.4f} | {h['arm_mae_deg']:.2f} | "
                     f"{h['score']:.3f}{mark} |")
    L += ["", f"Xem đồ thị: `tensorboard --logdir {out_dir}`", ""]
    with open(os.path.join(out_dir, "results.md"), "w") as fh:
        fh.write("\n".join(L))


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--open_loop_max", type=int, default=0)
    args, rest = ap.parse_known_args()
    cfg = parse_config(rest, description="Pipeline: cache -> train -> eval -> results.md. "
                                         "Thêm: --skip_train, --open_loop_max N")
    os.makedirs(cfg.out_dir, exist_ok=True)
    log = open(os.path.join(cfg.out_dir, "pipeline.log"), "a")
    sys.stdout, sys.stderr = Tee(sys.__stdout__, log), Tee(sys.__stderr__, log)
    print(f"\n[pipeline] ===== {time.strftime('%Y-%m-%d %H:%M:%S')} | {' '.join(sys.argv)}")
    for n in ("results.md", "results.json"):      # results.md chỉ tồn tại khi lần chạy này đã xong
        if os.path.exists(os.path.join(cfg.out_dir, n)):
            os.remove(os.path.join(cfg.out_dir, n))

    def on_term(signum, frame):                    # `kill <pid>` -> ghi STATUS thay vì chết im lặng
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, on_term)
    t_start = time.time()
    try:
        if not os.path.exists(os.path.join(cfg.cache_dir, "meta.json")):
            set_status(cfg.out_dir, "RUNNING: tạo cache")
            prepare(cfg)
        if not args.skip_train:
            set_status(cfg.out_dir, "RUNNING: train")
            train.main(cfg)
        set_status(cfg.out_dir, "RUNNING: eval")
        ckpt = next((p for p in (os.path.join(cfg.out_dir, n) for n in ("best.pt", "last.pt"))
                     if os.path.exists(p)), None)
        if ckpt is None:
            raise SystemExit(f"[pipeline] Không có best.pt / last.pt trong {cfg.out_dir}")
        res = final_eval(ckpt, cfg.out_dir, args.open_loop_max)
        write_report(cfg.out_dir, res, eval_history(cfg.out_dir), t_start)
        set_status(cfg.out_dir, f"DONE -> {os.path.join(cfg.out_dir, 'results.md')}")
    except KeyboardInterrupt:
        set_status(cfg.out_dir, "STOPPED (Ctrl+C / kill). Chạy lại cùng lệnh với --resume auto để tiếp tục")
        raise
    except BaseException as e:
        traceback.print_exc()
        set_status(cfg.out_dir, f"FAILED: {type(e).__name__}: {e} (chi tiết trong pipeline.log)")
        raise
    finally:
        sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
        log.close()


if __name__ == "__main__":
    main()
