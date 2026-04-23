# Improvement Ablation Report (Kickoff)

This report summarizes the first improvement-pass experiments using the synthetic smoke setup.

## Baseline Reference

- Reference checkpoint: `outputs/rsf_unet_drive_smoke/best.pt`
- Baseline config: `configs/drive_rsf_smoke.yaml`
- Baseline in-domain metric: `AUC=0.697349`

## Ablations Executed

1. `configs/ablation/bce_dice_smoke.yaml` (`loss=bce_dice`)
2. `configs/ablation/bce_focal_smoke.yaml` (`loss=bce_focal`)
3. `configs/ablation/vessel_sampling_smoke.yaml` (`vessel_sampling_prob=0.7`)
4. `configs/ablation/tta_postprocess_smoke.yaml` (eval-only TTA + threshold sweep + morphological postprocess)

## Results (Synthetic DRIVE Test)

- `BCE + Dice`: `AUC=0.711947`
- `BCE + Focal`: `AUC=0.697704`
- `Vessel-aware sampling`: `AUC=0.681837`
- `TTA + threshold + postprocess` on baseline checkpoint: no F1 gain in this smoke setup

## Best Config In This Round

- **Winner:** `BCE + Dice` (`configs/ablation/bce_dice_smoke.yaml`)
- **Improvement over baseline:** `+0.014598 AUC`
- Best checkpoint: `outputs/rsf_ablate_bce_dice_smoke/best.pt`

## Interpretation

- On this synthetic data, loss-level change (`BCE + Dice`) helped most.
- Sampling and postprocessing effects were neutral/negative in this small smoke setup.
- Real-data behavior can differ significantly; repeat the same ablations on true DRIVE/STARE/CHASE_DB1 before final conclusions.

## Next Real-Data Steps

1. Run `BCE` vs `BCE+Dice` on real DRIVE with full epochs.
2. Re-run OOD tests (`DRIVE -> STARE`, `DRIVE -> CHASE_DB1`) for the better loss.
3. Then test vessel-aware sampling and TTA/postprocessing again under real distributions.
