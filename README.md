# OpenArm VA flow-matching policy

DINOv2 + SigLIP (đóng băng) + depth CNN + qpos → context encoder → action expert (adaLN-Zero) → chunk 40×16, Euler 10 bước.
Tay dùng Δq, kẹp dùng giá trị tuyệt đối. Khoảng 68M tham số huấn luyện. Suy luận cần khoảng 2 GB VRAM.

## Cài đặt
```bash
pip install -r requirements.txt
```

## Dữ liệu `Tuyen062004/data_openarm_8_10`
Đã kiểm tra trực tiếp trên Hugging Face (2026-10-12):
- 85 episode, mỗi episode 14–22 s, ghi ở **50 Hz** (~69.500 frame). RGB 640×480 + depth uint16 (mm), 16 khớp.
- `action` là **bản sao của `observations/qpos`**, không phải lệnh. Lệnh thật là `joint_commands` (kẹp ra lệnh 0 / 0,043, còn qpos kẹp chỉ về ~0,024 khi kẹp vào vật). Luôn dùng `--key_cmd joint_commands`.
- 49–87 frame đầu mỗi episode có `joint_commands` = NaN (trước khi teleop gửi lệnh). `data.py` tự bỏ đoạn này.
- `--frame_stride 2` hạ 50 Hz → 25 Hz để chunk 40 = 1,6 s và robot chạy vòng 25 Hz như thiết kế. Sau khi hạ còn khoảng 32.000 frame, ~28.000 frame train.

## Một lệnh: tạo cache → train → eval → ghi kết quả
`pipeline.py` nhận mọi cờ của `data.py` và `train.py`. Nếu `cache_dir` chưa có thì tạo cache (tự tải dataset về `raw_dir` nếu thư mục trống). Sau đó train, rồi eval `best.pt` trên **toàn bộ** tập val (tách theo từng tay) và chạy open-loop trên từng episode val. Lệnh dưới đây đã gồm cấu hình đề xuất cho GPU 32 GB:
```bash
nohup python pipeline.py \
  --raw_dir data/raw --cache_dir data/cache --out_dir runs/vafm_b128 --seed 0 \
  --key_rgb observations/images/chest_rgb --key_depth observations/images/chest_depth \
  --key_cmd joint_commands --key_state observations/qpos \
  --frame_stride 2 --hz 25 --chunk 40 --depth_unit_m 0.001 --rgb_is_bgr false \
  --crop_xyxy "[104,48,536,480]" --cache_rgb_size 256 --cache_depth_size 160 \
  --arm_joints "[[0,1,2,3,4,5,6],[8,9,10,11,12,13,14]]" --grippers "[7,15]" \
  --epochs 80 --batch_size 128 --num_workers 8 \
  --lr 1.5e-4 --weight_decay 0.05 --warmup_steps 500 --grad_clip 1.0 --ema_decay 0.999 \
  --n_flow_samples 4 --inactive_weight 0.1 --gripper_weight 2.0 --balance_arms true \
  --split_mode cache --val_frac 0.12 --eval_samples 2000 --eval_every 0 --eval_every_epochs 2 \
  --log_every 20 --tensorboard true --save_every 0 --save_every_epochs 10 --keep_ckpts 3 --resume auto \
  --open_loop_max 0 \
  > /dev/null 2>&1 &
```
Lệnh chạy nền: tắt terminal hay mất SSH thì vẫn tiếp tục chạy. Sau đó chỉ cần xem:
```bash
cat runs/vafm_b128/STATUS          # RUNNING: tạo cache | train | eval  /  DONE  /  FAILED  /  STOPPED
cat runs/vafm_b128/results.md      # chỉ xuất hiện khi DONE
tail -f runs/vafm_b128/pipeline.log    # xem trực tiếp (Ctrl+C chỉ thoát tail, không dừng train)
tensorboard --logdir runs
```
- `results.md`: so sánh model với baseline đứng yên (toàn val và từng tay), sai số theo bước trong chunk, open-loop từng episode, lịch sử eval trong lúc train. Bản máy đọc nằm ở `results.json`.
- Bị ngắt (`STOPPED`) hoặc lỗi (`FAILED`, xem `pipeline.log`): chạy lại **đúng lệnh trên**. `--resume auto` sẽ train tiếp từ `last.pt`, cache đã có thì không tạo lại.
- Dừng hẳn: `pkill -f pipeline.py` (STATUS sẽ ghi `STOPPED`).
- Chỉ eval lại một lần train đã xong: thêm `--skip_train`. `--open_loop_max N` giới hạn số episode chạy open-loop (`-1` = bỏ qua).
- Nếu đã tạo cache bằng cấu hình khác (key sai, 50 Hz...) thì xoá `data/cache` trước khi chạy.

## Quy trình
```bash
# 0. Xem cấu trúc HDF5 và key được tự dò
python tools.py inspect data/raw/episode_20261008_152119.hdf5 \
  --key_rgb observations/images/chest_rgb --key_depth observations/images/chest_depth \
  --key_cmd joint_commands --key_state observations/qpos

# 1. Kiểm tra bố cục 16 khớp: tay hoạt động phải có biên độ lớn ở đúng 7 cột của nó
python tools.py joints --raw_dir data/raw \
  --key_rgb observations/images/chest_rgb --key_depth observations/images/chest_depth \
  --key_cmd joint_commands --key_state observations/qpos

# 2. Chọn vùng crop, mở crop_preview.png, chỉnh tới khi vật và kẹp to rõ ở hàng dưới
#    (nếu màu bị đảo xanh/đỏ thì thêm --rgb_is_bgr true vào cả lệnh này và bước 3)
python tools.py preview-crop --raw_dir data/raw --crop_xyxy "[104,48,536,480]" \
  --key_rgb observations/images/chest_rgb --key_depth observations/images/chest_depth \
  --key_cmd joint_commands --key_state observations/qpos

# 3. Tạo cache (~8 GB). Tự tải dataset về data/raw nếu thư mục trống
python data.py --raw_dir data/raw --cache_dir data/cache \
  --key_rgb observations/images/chest_rgb --key_depth observations/images/chest_depth \
  --key_cmd joint_commands --key_state observations/qpos \
  --frame_stride 2 --hz 25 --chunk 40 --depth_unit_m 0.001 --rgb_is_bgr false \
  --crop_xyxy "[104,48,536,480]" --cache_rgb_size 256 --cache_depth_size 160 \
  --arm_joints "[[0,1,2,3,4,5,6],[8,9,10,11,12,13,14]]" --grippers "[7,15]" \
  --val_frac 0.12 --seed 0

# 4. (tuỳ chọn) dùng trọng số thị giác của OpenVLA thay cho trọng số timm gốc
python tools.py extract-openvla --out weights/openvla_vision.safetensors

# 5. Train: lệnh ở mục "Lệnh train (GPU 32 GB)" bên dưới
tensorboard --logdir runs

# 6. Kiểm tra open-loop trên một episode val (tên episode val có trong <out_dir>/split.json)
python policy.py --ckpt runs/vafm_b128/best.pt --h5 data/raw/<episode_val>.hdf5
```

## Lệnh train (GPU 32 GB)
Cấu hình đề xuất cho dataset trên và GPU 32 GB. Copy chạy luôn:
```bash
python train.py \
  --cache_dir data/cache --out_dir runs/vafm_b128 --seed 0 \
  --epochs 80 --batch_size 128 --num_workers 8 \
  --lr 1.5e-4 --weight_decay 0.05 --warmup_steps 500 --grad_clip 1.0 --ema_decay 0.999 \
  --n_flow_samples 4 --inactive_weight 0.1 --gripper_weight 2.0 --balance_arms true \
  --split_mode cache --val_frac 0.12 --eval_samples 2000 --eval_every 0 --eval_every_epochs 2 \
  --log_every 20 --tensorboard true --tb_dir None \
  --save_every 0 --save_every_epochs 10 --keep_ckpts 3 --resume auto --max_steps_debug None \
  --dino_name vit_large_patch14_reg4_dinov2.lvd142m --siglip_name vit_so400m_patch14_siglip_224 \
  --backbone_pretrained true --openvla_vision_path None \
  --d_model 512 --n_heads 8 --enc_layers 4 --expert_layers 8 --mlp_ratio 4.0 --dropout 0.1 --flow_steps 10 \
  --img_size 224 --depth_size 128 --crop_scale_train "[0.84,1.0]" --crop_scale_eval 0.92 \
  --depth_min_m 0.2 --depth_max_m 1.5 \
  --aug_brightness 0.2 --aug_contrast 0.2 --aug_saturation 0.2 --aug_rgb_noise 0.02 \
  --state_noise_rad 0.01 --depth_noise_m 0.005 --depth_hole_prob 0.5
```
`--resume auto` để chạy lại đúng lệnh này khi bị ngắt: nếu `runs/vafm_b128/last.pt` có sẵn thì train tiếp, không thì train từ đầu.
Nếu làm bước 4 thì đổi `--openvla_vision_path None` thành `--openvla_vision_path weights/openvla_vision.safetensors`.

Các giá trị khác mặc định và lý do:

| Cờ | Mặc định → đề xuất | Lý do |
|---|---|---|
| `--batch_size` | 48 → 128 | VRAM ≈ 3,6 GB cố định + ~170 MB/mẫu (đã đo activation) → ~25 GB ở 128, trần ~160 trên 32 GB |
| `--lr` | 1e-4 → 1.5e-4 | tăng theo √(128/48) ≈ 1,6 khi batch lớn hơn |
| `--epochs` | None (40.000 bước) → 80 | ~28.000 frame train / 128 ≈ 220 bước/epoch → ~17.600 bước, ~2,2 triệu mẫu (gần bằng mặc định 1,9 triệu) |
| `--warmup_steps` | 1000 → 500 | giữ số mẫu warmup gần như cũ (~64.000 mẫu, ~2 epoch) |
| `--eval_every_epochs`, `--eval_samples` | 1, 1000 → 2, 2000 | val ~3.800 frame (10 episode). 2000 mẫu cho số đo ổn định hơn; eval mỗi 2 epoch tốn ~5% thời gian |
| `--save_every`, `--save_every_epochs` | 2000, 1 → 0, 10 | `last.pt` vẫn ghi cuối mỗi epoch. Mỗi file ~1,4 GB (gồm optimizer), nên chỉ giữ 3 mốc mỗi 10 epoch |
| `--log_every` | 50 → 20 | mỗi epoch chỉ ~220 bước |

Kiểm tra trước khi chạy dài: chạy 60 bước rồi xem `perf/gpu_mem_gb` (chưa tính ~0,8 GB CUDA context) và `perf/s_per_it` trong TensorBoard. Nếu dưới ~26 GB thì có thể tăng lên `--batch_size 160 --lr 1.8e-4`; nếu bị OOM thì dùng `--batch_size 96 --lr 1.3e-4`.
```bash
python train.py --cache_dir data/cache --out_dir runs/memtest --batch_size 128 --max_steps_debug 60 --log_every 20 --eval_every_epochs 0
```
Theo dõi: `loss/val` và `val/score` tăng trở lại trong khi `loss/train` vẫn giảm là dấu hiệu overfit; `best.pt` đã tự giữ mốc tốt nhất.

Lệnh hay dùng khác:
```bash
# chia lại 20% episode làm val mà không cần tạo lại cache
python train.py --cache_dir data/cache --out_dir runs/vafm_val20 --epochs 80 --batch_size 128 --lr 1.5e-4 --warmup_steps 500 --split_mode resplit --val_frac 0.2

# chạy thử nhanh 20 bước, eval mỗi 10 bước
python train.py --cache_dir data/cache --out_dir runs/debug --max_steps_debug 20 --eval_every 10 --eval_samples 96
```
Khi resume, dùng lại đúng các cờ của lần chạy trước (nhất là `--epochs`, `--batch_size`, `--split_mode`, `--val_frac`, `--seed`). Cờ có giá trị `None` hoặc `true/false` đều viết được như trên; cờ bool viết trơn (`--tensorboard`) nghĩa là `true`. Các cờ về cache (`--crop_xyxy`, `--chunk`, `--arm_joints`, ...) bị bỏ qua khi train vì luôn lấy theo `meta.json` của cache.

| Cờ | Ý nghĩa |
|---|---|
| `--cache_dir`, `--out_dir` | thư mục cache (bước 3) và thư mục lưu kết quả |
| `--seed` | seed chung: khởi tạo, chia train/val, chọn tập con eval |
| `--epochs` | số epoch. `None` = dùng `--total_steps`. Đặt số thì ghi đè `total_steps = epochs × số bước/epoch` |
| `--total_steps` | tổng số bước train khi `--epochs None` |
| `--batch_size`, `--num_workers` | kích thước batch, số tiến trình đọc dữ liệu |
| `--lr`, `--weight_decay`, `--warmup_steps`, `--grad_clip`, `--ema_decay` | AdamW, warmup tuyến tính rồi cosine về 0.1×lr, cắt gradient, EMA trọng số |
| `--n_flow_samples` | số cặp (t, nhiễu) trên mỗi quan sát (backbone chạy 1 lần) |
| `--inactive_weight`, `--gripper_weight` | trọng số loss tay đứng yên / kẹp tay làm việc |
| `--balance_arms` | lấy mẫu cân bằng episode tay 0 / tay 1 |
| `--split_mode` | `cache` = dùng split trong `meta.json`; `resplit` = chia lại theo episode bằng `--val_frac`, `--seed` |
| `--val_frac` | tỉ lệ episode làm val (0 = không eval) |
| `--eval_samples` | số mẫu val cố định để eval (0 = toàn bộ val) |
| `--eval_every`, `--eval_every_epochs` | eval mỗi N bước / mỗi N epoch (0 = tắt) |
| `--log_every` | in và ghi log train mỗi N bước |
| `--tensorboard`, `--tb_dir` | bật/tắt TensorBoard; thư mục log (`None` = `<out_dir>/tb`) |
| `--save_every` | ghi đè `last.pt` mỗi N bước (0 = chỉ cuối epoch) |
| `--save_every_epochs`, `--keep_ckpts` | lưu `epoch_XXXX.pt` mỗi N epoch, giữ N file mới nhất (0 = giữ hết) |
| `--resume` | đường dẫn checkpoint, `auto` = `<out_dir>/last.pt` nếu có |
| `--max_steps_debug` | chỉ chạy N bước để thử |
| `--dino_name`, `--siglip_name`, `--backbone_pretrained`, `--openvla_vision_path` | backbone thị giác; đặt `--openvla_vision_path weights/openvla_vision.safetensors` nếu làm bước 4 |
| `--d_model`, `--n_heads`, `--enc_layers`, `--expert_layers`, `--mlp_ratio`, `--dropout`, `--flow_steps` | kích thước model; số bước Euler khi lấy mẫu |
| `--img_size`, `--depth_size`, `--crop_scale_train`, `--crop_scale_eval` | kích thước đầu vào; tỉ lệ random crop khi train / crop giữa khi eval |
| `--depth_min_m`, `--depth_max_m` | khoảng clip depth (m) |
| `--aug_*`, `--state_noise_rad`, `--depth_noise_m`, `--depth_hole_prob` | augmentation màu, nhiễu qpos, nhiễu và lỗ giả trên depth |

Trong `out_dir`: `last.pt` (để resume), `epoch_XXXX.pt`, `best.pt` (score eval tốt nhất, dùng cho suy luận), `config.json`, `split.json` (episode train/val), `log.jsonl`, `tb/`.

## Xử lý 6 rủi ro
| Rủi ro | Ở đâu | Cách xử lý |
|---|---|---|
| 1. Độ phân giải | `data.preprocess_frame`, `tools.py preview-crop` | Crop vùng làm việc trên ảnh gốc rồi mới resize. Dùng chung code cho cache và robot |
| 2. Mất augmentation | `data.gpu_preprocess`, `train.py` | Không cache feature. Backbone chạy online (bf16, no_grad). Random crop chung cho RGB+depth, color jitter. `n_flow_samples=4` để bù chi phí backbone |
| 3. Chuẩn hoá Δq | `data.compute_stats`, buffer `action_mean/std (40,16)` | Mean/std riêng cho từng (bước, khớp), chỉ tính trên tay đang làm việc |
| 4. Tay không hoạt động | `train.masks_for`, `loss_weights`, `evaluate` | Tự nhận tay hoạt động mỗi episode. Trọng số loss tay đứng yên 0.1, kẹp 2.0. Log riêng `mse_arm/grip/idle`. Eval bằng độ so với baseline đứng yên |
| 5. Depth nhiễu | `data.to_meters`, `gpu_preprocess`, `model.DepthEncoder` | Clip 0.2–1.5 m, kênh mask, nội suy chuẩn hoá theo mask, nhiễu và lỗ giả khi train. CNN chỉ ~0.5M tham số |
| 6. Ít dữ liệu | toàn bộ | Backbone đóng băng, augmentation, nhiễu qpos, dropout, EMA, chia val theo episode và phân tầng theo tay, lấy mẫu cân bằng hai tay, lưu best theo chỉ số val |

## Đọc log
- `mse_idle` thấp là bình thường, không có ý nghĩa. Hãy theo dõi `mse_arm` và `mse_grip`.
- Chỉ số quan trọng nhất: `arm_mae_deg` phải thấp hơn rõ rệt so với `hold_mae_deg` (baseline đứng yên). Để ý `arm_mae_deg@39`.
- Nếu `grip_mae` không thấp hơn `grip_hold` thì model chưa học được lúc nào đóng kẹp.

## Trên robot
```python
from policy import VAPolicy, ChunkExecutor
ex = ChunkExecutor(VAPolicy("runs/vafm/best.pt"), mode="chunk", exec_horizon=20)
q_target = ex.step(rgb, depth, qpos)   # gọi mỗi chu kỳ 25 Hz, gửi q_target cho robot
```
Ảnh truyền vào phải cùng độ phân giải, thứ tự kênh và căn chỉnh depth với lúc ghi dữ liệu.
