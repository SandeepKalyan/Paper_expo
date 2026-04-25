# RSF-Conv talk — speech script

_Auto-generated on 2026-04-24 18:09 by `scripts/build_deck.py`. Re-run after every deck rebuild._

**Target duration:** 12 min talk + 3 min Q&A in a 15-min slot.

**Rule:** slide holds the headline; script holds the explanation. Do **not** read the slide.

---

**Slide 01 — Title (30 s).** RSF-Conv from IEEE TNNLS 2025 (Sun et al.). Goal: faithful reproduction of DRIVE retinal-vessel numbers, then beat them. Walkthrough: idea → reproduction → numbers → lessons.

**Slide 02 — Problem (60 s).** Vessels matter clinically; annotation is manual; benchmarks are small. Core insight: vessels have no canonical rotation or scale, so the conv shouldn't either. Standard conv breaks that symmetry; RSF-Conv preserves it.

**Slide 03 — Paper claims (45 s).** Three headline numbers: 13.9% of U-Net's params, 0.9816 DRIVE AUC, 0.8270 F1. Target numbers repeat across three benchmarks, listed at the bottom. My reproduction aims at these.

**Slide 04 — Operator (75 s).** Standard conv learns pixel weights. RSF-Conv learns Fourier coefficients over rotation n and scale s, synthesises 32 oriented+scaled kernels at forward time, then group-pools over rotation. Outcome: rotation equivariance, scale invariance, ~7× parameter saving.

**Slide 05 — Architecture (45 s).** Faithful paper U-Net. 4 encoder levels, bottleneck, 4 decoder levels, skip connections. Every conv inside is RSF-Conv. 1.68 M params total. Engineering on top: AMP, cosine LR, weight decay, sliding-window TTA — no architectural changes.

**Slide 06 — Curves (45 s).** Val AUC left, F1 right. Dashed orange = paper target. Laptop run hits ~0.979 AUC by ep 8 then crashes. A100 seed-43 still climbing. Eval-side ensembling closes the remaining gap.

**Slide 07 — DRIVE (75 s).** Paper vs best single vs ensemble across 5 metrics. Strip below chart shows the two deltas that matter: best-single ΔAUC and ensemble ΔF1. Read the strip aloud.

**Slide 08 — OOD (60 s).** Train DRIVE, test STARE + CHASE. No fine-tuning. Paper bars vs our ensemble bars. Close match = claim holds. Big gap = we overfit DRIVE.

**Slide 09 — Qualitative (60 s).** Two DRIVE cases, four panels each. Best case (AUC ≈ 0.99) at top, median case below. Error map: red = false positives, blue = missed vessels, white = correct. Most residual errors are capillary tips — same failure mode the paper notes.

**Slide 10 — Takeaways (45 s).** Reproduces · beats via eval tricks · transfers out-of-domain. Next-week lever: 200-epoch schedule + Dice/Focal ablation + 5-seed ensemble.

**Slide 11 — Q&A (as needed).** Likely questions + answers prepared in speaker notes. Keep responses short.

---

## Q&A prep

1. **Why cosine LR, not constant?** bs-4 + constant LR showed late-epoch degradation; cosine removed it empirically.
2. **How much does TTA add?** ~+0.003 F1, ~+0.002 AUC in my ablations — non-trivial at this precision.
3. **Why only 50 epochs vs paper's 200?** Compute budget — A100 for ~6 h; 50 ep at bs 4 filled it. Longer training is the obvious next lever.
4. **Why bs=4 not paper's bs=2?** bs=2 under-utilised the A100; bs=4 with cosine LR + wd was the best speed/quality trade-off I measured.
5. **Why not Dice / Focal loss?** Paper used BCE; I didn't want to conflate the operator contribution with a loss change. Queued for ablation.
6. **Is the ensemble 'fair'?** It uses only DRIVE-train checkpoints. No STARE/CHASE data was used for ensembling — those are strict OOD.
7. **How are F1/AUC computed?** Inside the FOV mask only, per-image, then averaged — matches the paper's protocol.
8. **What would you do with another week?** (a) 200-epoch schedule for each seed, (b) explicit scale augmentation, (c) Dice/Focal ablation, (d) 5-seed ensemble.
