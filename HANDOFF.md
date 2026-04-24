# Project Handoff — 2026-04-24 ~03:40 PDT

## CRITICAL — Deadline

- **Submission: Saturday 10:00 IST = Fri 2026-04-24 21:30 PDT**
- As of writing: **~18h remaining**
- Deliverables: (1) beat paper numbers on retinal vessel seg, (2) presentation
- Project: reproduce "RSF-Conv" (IEEE TNNLS 2025, Sun et al.). Paper at `Base_paper.pdf`

## Paper Target Numbers (to beat)

**In-domain Table I — RSF-Conv + U-Net:**
| Dataset | Se | Sp | F1 | Acc | AUC |
|---|---|---|---|---|---|
| DRIVE | 79.95 | 98.05 | 82.70 | 95.74 | **98.16** |
| STARE | 80.02 | 98.82 | 83.18 | 97.19 | 98.99 |
| CHASE_DB1 | 81.26 | 98.26 | 81.79 | 96.72 | 98.74 |

**OOD Table II — RSF-Conv + U-Net:**
| Transfer | Se | Sp | F1 | Acc | AUC |
|---|---|---|---|---|---|
| DRIVE→STARE | 72.73 | 99.02 | 79.45 | 96.74 | 97.95 |
| DRIVE→CHASE_DB1 | 76.79 | 96.87 | 73.76 | 95.05 | 96.44 |

**Paper training:** Adam lr=2e-4, batch=2, 200 epochs, patch 256×256, BCE only, augs = rotation+rescale+flip+shear+brightness+saturation+contrast. Test = overlapping 256×256 stride 128 patches, threshold 0.5. Metrics inside FOV.

**Paper RSFConv hyperparams:** `p=6 h=0.5`, rotations `{iπ/4, i=0..7}` (rotNum=8), scales `{(5/4)^i, i=0..3}` (scaleNum=4, μ=1.25). Weight sharing reduces params to 13.9% of backbone (4.32M for RSF-Conv+U-Net vs 31.04M standard U-Net).

## Current Training State

### A100 Pod (active)
- **vast.ai instance #35208878**, Quebec, $0.602/hr
- **SSH:** `ssh -i ~/.ssh/vast_ai -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 23833 root@108.231.141.46`
- Private key: `C:\Users\vvsvs\.ssh\vast_ai` (ed25519, no passphrase)
- **Running now:** `rsf_paper_proper` (proper paper RSFConv port). Started 2026-04-24 ~03:20 PDT. Epoch 2/200 at ~03:25.
- **ETA:** ~3h (60s/epoch × 200 epochs) → ~06:20 PDT
- Working dir on pod: `/workspace/Paper_expo`
- Data dir on pod: `/workspace/Paper_expo/data/{DRIVE,STARE,CHASE_DB1,HRF,raw}/` — already prepared
- Python env: `/workspace/Paper_expo/.venv/` (torch 2.11.0+cu130)

### Laptop 3080 (can idle)
- Windows 11, Python 3.13, venv at `.venv/`
- Baseline UNet on DRIVE ran to ep57 then stalled (process died), best AUC 0.9736 @ ep15 in `outputs/unet_drive_real/`. Overfit after ep15. Can be ignored or deleted.
- No active GPU process. Free to launch ablations if wanted.

### Cron Poll Job
- Cron ID `0e5247fe` polls A100 every 20 min
- Dies when Claude session ends. User should renew with `/loop 20m ...` if needed

## Results Recorded

### Completed on A100 (in `/workspace/Paper_expo/outputs/`):
1. **`rsf_unet_paper/`** — NAIVE RSFConv sketch, BCE, paper augs, 200 epochs
   - Best ep15: **AUC=0.9349 F1=0.7148 Se=0.6423 Sp=0.9774**
   - Severe overfit after ep15 (train_loss 0.008 at ep200, val AUC 0.9101)
2. **`rsf_unet_plus/`** — NAIVE RSFConv + BCE+Dice + vessel-aware sampling + bigger batch
   - Killed at ep~80 to free GPU for proper run
   - Best ep10: **AUC=0.9555 F1=0.7741 Se=0.7245 Sp=0.9785**
3. **`rsf_paper_proper/`** — ACTIVE. True paper G-CNN port. In progress.

### Sister's DRIVE baseline (external reference, `per_image_results.xlsx` on Desktop):
- 20 DRIVE test images: mean AUC=0.9694, F1=0.7264, Se=0.8774, Sp=0.9488, Acc=0.9424, CLDICE=0.7591, HD95=3.31px
- Sister probably ran vanilla U-Net or similar setup. Proves achievable target.

## Code State — Branch `laptop/src-data-and-rsf-opts`

**Latest commit: `538ba11`** on branch `laptop/src-data-and-rsf-opts` (NOT main — PR not merged yet due to push protection on main).

### Key Files Added/Modified

```
src/data/                       # NEW — dataset + loaders (was missing)
  __init__.py
  dataset.py                    # RetinalDataset + DriveDataset/StareDataset/ChaseDB1Dataset/HRFDataset
  factory.py                    # build_dataloader(cfg, split, train)
  sampler.py                    # random_patch_coords, vessel_centered_patch_coords
  transforms.py                 # augment_train with use_paper_augs flag (rotation/rescale/shear/color)
src/models/rsf_conv.py          # MODIFIED — optimized sketch (5.8× speedup via batched grid_sample)
src/models/rsf_paper/           # NEW — ported from szhc0gk/RSF-Conv
  __init__.py                   # exports RSFConvUnet
  rsf_conv_paper.py             # RSFconv + Fourier basis + GroupPooling + RSF_BN
  parts.py                      # DoubleConv/Down/Up/InConv/OutConv
  unet.py                       # RSFConvUnet (return_logits=True for BCEWithLogitsLoss)
src/models/factory.py           # MODIFIED — registers rsf_paper
src/patch_eval.py               # NEW — predict_patches_overlap (sliding window stride 128)
src/eval.py                     # MODIFIED — patch_overlap + TTA + threshold sweep via config
src/train.py                    # MODIFIED — bf16/fp16 autocast, metrics.jsonl streaming, last.pt ckpt
scripts/a100_bootstrap.sh       # NEW — one-shot pod setup
scripts/a100_run_all.sh         # NEW — original queue (superseded)
scripts/a100_run_proper.sh      # NEW — proper paper RSFConv queue (CURRENT)
scripts/track_remote.sh         # NEW — SSH tracking helper
configs/drive_baseline.yaml     # MODIFIED — use_paper_augs: true
configs/drive_rsf.yaml          # MODIFIED — k=6 base=16 default
configs/drive_rsf_plus.yaml     # NEW — plus config
configs/a100/                   # NEW — all A100-tuned configs (bf16)
  drive_baseline.yaml
  drive_rsf_paper.yaml          # NAIVE RSFConv paper config
  drive_rsf_paper_proper.yaml   # TRUE paper port config (bs=32, lr=4e-4 for A100 throughput)
  drive_rsf_plus.yaml           # NAIVE + tricks
  stare_rsf.yaml
  chase_rsf.yaml
.gitignore                      # MODIFIED — anchored to /data/ /outputs/ (was blocking src/data/)
```

### Known Impl Details
- Our `RSFConv2d` in `src/models/rsf_conv.py` is a SIMPLIFIED sketch (random iRFFT init, averages 32 transforms). Underperforms. Retained for ablation comparison.
- `src/models/rsf_paper/rsf_conv_paper.py` is the TRUE paper port from github.com/szhc0gk/RSF-Conv (MIT-ish implicit). Has proper G-CNN group structure: feature maps have `rotNum*scaleNum=32` extra channels; filters stored as Fourier coefficients + precomputed bases; cyclic-shifted for equivariance.
- Train always uses BCEWithLogitsLoss or bce_dice/bce_focal (see `src/losses.py`). Paper port was modified with `return_logits=True` to strip their final `F.sigmoid`.
- Val metrics computed inside FOV mask (paper convention).
- Val: full image padded to /32 multiple (default). Optional: `patch_overlap: true` in cfg → sliding 256×256 patches stride 128.

## Pipeline Smoke Check

Run any time on laptop with GPU:
```bash
.venv/Scripts/python.exe -c "
from src.models.factory import build_model
from src.data.factory import build_dataloader
import torch
cfg_m = {'name':'rsf_paper','in_channels':3,'out_channels':1}
m = build_model(cfg_m).cuda()
cfg_d = {'name':'drive','root':'data/DRIVE','train_split':'train','patch_size':[256,256],'samples_per_epoch':2,'batch_size':2,'num_workers':0,'pin_memory':False}
dl = build_dataloader(cfg_d, 'train', True)
b = next(iter(dl))
o = m(b['image'].cuda())
print(o.shape, 'logits range', float(o.min()), float(o.max()))
"
```

Expected: `torch.Size([2, 1, 256, 256]) logits range ~ -2..2`.

## A100 Pod Ops — Copy-Paste Reference

### Quick status poll
```bash
ssh -i ~/.ssh/vast_ai -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 23833 root@108.231.141.46 "cd /workspace/Paper_expo && nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader && echo '---' && for f in outputs/*/metrics.jsonl; do echo \"### \$f\"; tail -3 \"\$f\"; done && for d in outputs/*/; do [ -f \$d/metrics.jsonl ] || continue; n=\$(wc -l < \$d/metrics.jsonl); best=\$(python3 -c \"import json; h=[json.loads(l) for l in open('\$d/metrics.jsonl')]; b=max(h,key=lambda x:x['AUC']); print(f'ep{b[\\\"epoch\\\"]} AUC={b[\\\"AUC\\\"]:.4f} F1={b[\\\"F1\\\"]:.4f}')\" 2>/dev/null); echo \"\$d n=\$n best=\$best\"; done"
```

### Pull latest code + resume queue
```bash
ssh -i ~/.ssh/vast_ai -p 23833 root@108.231.141.46 "cd /workspace/Paper_expo && git pull && nohup bash scripts/a100_run_proper.sh > outputs/logs/proper_queue.log 2>&1 &"
```

### Kill all training
```bash
ssh -i ~/.ssh/vast_ai -p 23833 root@108.231.141.46 "pkill -9 -f 'src.train'; pkill -9 -f 'a100_run'"
```

### Download final artifacts to laptop
```bash
# Run from laptop
scp -i ~/.ssh/vast_ai -P 23833 -r root@108.231.141.46:/workspace/Paper_expo/outputs/ ./outputs_a100/
```

### Stop pod (END BILLING)
- Go to vast.ai dashboard → instance → **Destroy**
- Double-check before destroying. All `outputs/` need to be pulled first or lost.

## Priority Next Steps (ordered)

### P0 — Must finish before deadline
1. **Wait for `rsf_paper_proper` to finish** (~06:20 PDT if started 03:20). Check metrics.jsonl every 20 min via cron job (or manual SSH).
2. **Verify final AUC matches paper** (target 0.98+). If AUC < 0.97 by ep50, consider:
   - Decrease LR to 2e-4 (current 4e-4 may be too aggressive for bs=32)
   - Or run a bs=8 lr=2e-4 variant in parallel (~5h)
3. **OOD evals** (auto-queued after training, in `scripts/a100_run_proper.sh`): DRIVE→STARE, DRIVE→CHASE. Uses `best.pt`.
4. **Pull all artifacts to laptop** via `scp` before destroying pod.
5. **Build presentation** — see "Demo Materials" below.

### P1 — High-value additions if time allows
- **Add CLDICE + HD95 metrics** to match sister's report format. Implementation sketch:
  ```python
  # In src/metrics/vessel_metrics.py, add:
  from skimage.morphology import skeletonize
  from scipy.ndimage import distance_transform_edt
  def cldice(pred, gt):
      sp = skeletonize(pred); sg = skeletonize(gt)
      tprec = (sp*gt).sum()/(sp.sum()+1e-8); tsens = (sg*pred).sum()/(sg.sum()+1e-8)
      return 2*tprec*tsens/(tprec+tsens+1e-8)
  def hd95(pred, gt):
      # use scipy directed_hausdorff on both directions, take 95th percentile
      ...
  ```
- **In-domain STARE + CHASE_DB1 training** — paper has these too. `configs/a100/stare_rsf.yaml`, `chase_rsf.yaml` ready but use NAIVE `rsf_unet` model. Swap to `rsf_paper` and rerun.
- **Plot training curves** from `metrics.jsonl` for demo slides.

### P2 — Optional
- Retrain `rsf_unet_plus` with proper paper RSFConv backbone + BCE+Dice + TTA combo. Strongest single config to beat paper. ~3-4h additional.
- Ensemble of 3 random seeds on RSFConv paper proper. ~10h combined.

## Demo Materials (Presentation Checklist)

Build a 10-min slide deck with:
1. **Problem** (1 slide) — retinal vessel seg, clinical significance, domain shift
2. **Paper summary** (1 slide) — RSF-Conv = rotation+scale equivariant Fourier conv; plug-and-play UNet replacement; 13.9% params; in/out-of-domain wins
3. **Our pipeline** (1 slide) — data prep (deepdyn + HRF), EDA findings, training harness (bf16 AMP, metrics.jsonl streaming, last+best ckpts, paper augs, patch-stride eval, TTA)
4. **Implementation journey** (1 slide) — sketch RSFConv → debugging overfit → ported true paper G-CNN → final
5. **Results table** (1 slide):
   | Method | DRIVE AUC | DRIVE F1 | Notes |
   |---|---|---|---|
   | Paper U-Net | 0.9808 | 0.8243 | reported |
   | Paper RSF-Conv+U-Net | 0.9816 | 0.8270 | reported |
   | Sister's reproduction | 0.9694 | 0.7264 | external |
   | Our naive RSFConv | 0.9349 | 0.7148 | sketch impl |
   | Our naive + tricks | 0.9555 | 0.7741 | BCE+Dice+vessel |
   | **Our proper RSFConv** | TBD | TBD | target ≥ 0.9816 |
6. **OOD results** (1 slide) — DRIVE→STARE, DRIVE→CHASE
7. **EDA visuals** (1 slide) — domain shift PCA/t-SNE from `results/figs/`, per-dataset CLAHE previews
8. **Qualitative samples** (1 slide) — 2-3 test images with GT vs pred side-by-side (need to generate — see snippet below)
9. **Ablation ladder** (1 slide) — sketch RSF → + BCE+Dice → + vessel sampling → proper G-CNN. Shows our systematic path
10. **Takeaways** (1 slide) — what worked, what didn't, honest limitations

### Qualitative viz snippet (add to `scripts/visualize.py`)
```python
# Load best ckpt, run a few test images, save side-by-side PNG
# import torch, cv2, numpy as np
# from src.models.factory import build_model
# from src.data.factory import build_dataloader
# cfg = yaml.safe_load(...)
# m = build_model(cfg['model'])
# ckpt = torch.load('outputs/rsf_paper_proper/best.pt')
# m.load_state_dict(ckpt['model_state_dict'])
# ...generate overlay images
```

## Known Risks & Mitigations

1. **A100 proper run may OOM** at bs=32 during eval phase (full-image pass). Mitigation: if crashes, add `eval_batch_size: 1` to config.
2. **Proper RSFConv slow**: ~60s/epoch × 200 = 3h. If need faster: reduce epochs to 100 + early stop on val AUC plateau.
3. **vast.ai instance could get preempted** (rare, 99.78% reliability). Mitigation: `last.pt` saved every epoch; can resume.
4. **Results still below paper**: our port uses `factor_channels=2` (→ 1.68M params vs paper's 4.32M). If proper run still misses 0.98, boost channel count:
   - Edit `src/models/rsf_paper/unet.py` line 19: `factor_channels = ceil(64 / rotScaleNum)` → try `= 4` (3.36M params, closer to paper).
5. **Laptop CRLF/LF warnings** on every git commit — cosmetic, ignore.

## Files Outside Git (Don't Lose)

- `C:\Users\vvsvs\.ssh\vast_ai` — private SSH key. Needed to reach pod.
- `C:\Users\vvsvs\.ssh\vast_ai.pub` — pub key (already installed on vast.ai account).
- `C:\Users\vvsvs\Desktop\DRIVE CHASE STARE DATASET.zip` — sister's data. Not needed for training (we use deepdyn bundle). But sister's labels slightly different — don't mix.
- `C:\Users\vvsvs\Desktop\per_image_results.xlsx` — sister's DRIVE results (reference baseline 0.9694 AUC).
- `C:\Users\vvsvs\Desktop\Paper_expo\paper_text.txt` / `paper_ascii.txt` — pypdf-extracted paper text. Useful to grep. Not in git.

## Branches & Pushing

- Current branch: `laptop/src-data-and-rsf-opts`
- Direct push to `main` BLOCKED (protection). Always push to feature branch.
- To push: `git push origin HEAD:laptop/src-data-and-rsf-opts`
- On pod: `git checkout laptop/src-data-and-rsf-opts && git pull`
- To merge to main: open PR on GitHub.

## User Preferences

- **Caveman mode** is on. Terse responses. Drop articles, fluff, hedging. Keep technical substance.
- **Auto mode** is on. Execute without asking for approval on low-risk work.
- User wants documentation-heavy outputs for demo — always log structured metrics, save all ckpts + configs.
- User is OK spending up to $15 on cloud. Current burn: ~$0.60 × 3h = $1.80 so far.

## Timeline of This Session

- 23:30 PDT Thu: cloned repo, identified missing `src/data/`
- 00:30: wrote `src/data/` package (datasets, transforms, sampler, factory)
- 01:00: optimized RSFConv (5.8× speedup), patch_overlap eval, paper augs
- 01:30: kicked off laptop baseline UNet (ran to ep57, overfit, stalled)
- 02:00: rented vast.ai A100, SSH setup, bootstrap
- 02:30: launched queue (rsf_unet_paper, rsf_unet_plus)
- 03:00: rsf_unet_paper finished (0.9349), rsf_unet_plus hit 0.9555 by ep10 then degraded by ep80
- 03:20: ported paper's RSFConv from github.com/szhc0gk/RSF-Conv, killed old queue, launched proper run
- Now: proper run at ep2/200, ETA 3h
