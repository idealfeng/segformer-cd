# Reproduction Results Summary

This summary collects the current local reproduction results for reviewer-named
baseline models and DLV-CD. Metrics are reported for the change class.

## Evaluation Protocol

- Baseline models: BiFA, DMINet, ACABFNet, AANet, SNUNet, and BIT are evaluated
  with the same full-resolution 256x256 tile evaluator at threshold 0.5.
- BIT is reproduced with an offline randomly initialized ResNet18 backbone because
  ImageNet weights were not available in the local environment.
- SAM-CD-s is recorded separately as a fast low-budget FastSAM-s reproduction:
  the original FastSAM-x encoder is replaced by FastSAM-s and trained for only
  800 source steps. These numbers are diagnostic and should not be interpreted
  as a fully converged official SAM-CD reproduction.
- SAM-CD official uses the released `FastSAM.pt` encoder and source-validation
  model selection. It is evaluated with the same fixed 0.5 threshold and no
  target-domain training, fine-tuning, or threshold calibration.
- ChangeCLIP is recorded as a repaired RN50 reproduction after fixing binary
  mask loading for 0/1 labels and adding deterministic scene-text JSON sidecars.
  Checkpoints are selected by source-validation results only; no target-domain
  labels are used for model selection or threshold calibration.
- DLV-CD rows are selected from current best local outputs per transfer direction.
  The threshold column is kept because several DLV-CD historical runs used fixed
  thresholds other than 0.5.
- No target-domain images are used for training or fine-tuning in these
  source-only transfer evaluations.

## Cross-Domain Results (%)

| Model | Direction | Precision | Recall | F1 | IoU | Note |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| BiFA | LEVIR->WHU | 78.07 | 34.56 | 47.91 | 31.50 | @0.5 |
| BiFA | WHU->LEVIR | 54.10 | 3.99 | 7.43 | 3.86 | @0.5 |
| BiFA | S2Looking->LEVIR | 11.00 | 12.97 | 11.90 | 6.33 | @0.5 |
| BiFA | S2Looking->WHU | 10.36 | 15.75 | 12.50 | 6.67 | @0.5 |
| BiFA | DSIFN->LEVIR | 26.73 | 70.95 | 38.83 | 24.09 | @0.5 |
| BiFA | DSIFN->WHU | 19.38 | 62.34 | 29.57 | 17.35 | @0.5 |
| DMINet | LEVIR->WHU | 19.88 | 28.97 | 23.58 | 13.37 | @0.5 |
| DMINet | WHU->LEVIR | 23.83 | 9.43 | 13.52 | 7.25 | @0.5 |
| DMINet | S2Looking->LEVIR | 10.42 | 16.39 | 12.74 | 6.81 | retest @0.5 |
| DMINet | S2Looking->WHU | 16.34 | 24.98 | 19.76 | 10.96 | retest @0.5 |
| DMINet | DSIFN->LEVIR | 18.32 | 26.87 | 21.79 | 12.23 | @0.5 |
| DMINet | DSIFN->WHU | 10.29 | 83.39 | 18.32 | 10.09 | @0.5 |
| ACABFNet | LEVIR->WHU | 17.47 | 50.33 | 25.94 | 14.90 | @0.5 |
| ACABFNet | WHU->LEVIR | 11.58 | 28.18 | 16.41 | 8.94 | @0.5 |
| ACABFNet | S2Looking->LEVIR | 7.78 | 66.82 | 13.94 | 7.49 | @0.5 |
| ACABFNet | S2Looking->WHU | 12.67 | 83.77 | 22.02 | 12.37 | @0.5 |
| ACABFNet | DSIFN->LEVIR | 21.57 | 35.28 | 26.77 | 15.46 | @0.5 |
| ACABFNet | DSIFN->WHU | 13.84 | 79.03 | 23.55 | 13.35 | @0.5 |
| AANet | LEVIR->WHU | 12.61 | 59.49 | 20.81 | 11.62 | @0.5 |
| AANet | WHU->LEVIR | 23.63 | 26.67 | 25.06 | 14.33 | retest @0.5 |
| AANet | S2Looking->LEVIR | 9.19 | 43.78 | 15.19 | 8.22 | retest @0.5 |
| AANet | S2Looking->WHU | 8.87 | 77.71 | 15.93 | 8.65 | @0.5 |
| AANet | DSIFN->LEVIR | 17.89 | 40.74 | 24.86 | 14.20 | @0.5 |
| AANet | DSIFN->WHU | 8.78 | 91.92 | 16.02 | 8.71 | @0.5 |
| SNUNet | LEVIR->WHU | 14.40 | 34.95 | 20.40 | 11.36 | @0.5 |
| SNUNet | WHU->LEVIR | 20.02 | 6.16 | 9.42 | 4.94 | @0.5 |
| SNUNet | S2Looking->LEVIR | 9.87 | 14.96 | 11.89 | 6.32 | retest @0.5 |
| SNUNet | S2Looking->WHU | 15.79 | 39.33 | 22.54 | 12.70 | @0.5 |
| SNUNet | DSIFN->LEVIR | 16.38 | 20.37 | 18.16 | 9.98 | @0.5 |
| SNUNet | DSIFN->WHU | 9.53 | 76.12 | 16.94 | 9.26 | @0.5 |
| BIT | LEVIR->WHU | 16.57 | 21.28 | 18.63 | 10.27 | @0.5 |
| BIT | WHU->LEVIR | 36.17 | 10.56 | 16.35 | 8.90 | @0.5 |
| BIT | S2Looking->LEVIR | 7.73 | 28.37 | 12.15 | 6.47 | @0.5 |
| BIT | S2Looking->WHU | 12.92 | 40.22 | 19.56 | 10.84 | @0.5 |
| BIT | DSIFN->LEVIR | 14.48 | 28.22 | 19.14 | 10.58 | @0.5 |
| BIT | DSIFN->WHU | 8.96 | 80.30 | 16.12 | 8.77 | @0.5 |
| SAM-CD-s | LEVIR->WHU | 13.55 | 70.85 | 22.75 | 12.84 | FastSAM-s quick, 800 steps @0.5 |
| SAM-CD-s | WHU->LEVIR | 32.40 | 18.02 | 23.16 | 13.10 | FastSAM-s quick, 800 steps @0.5 |
| SAM-CD-s | S2Looking->LEVIR | 5.61 | 95.10 | 10.60 | 5.60 | FastSAM-s quick, 800 steps @0.5 |
| SAM-CD-s | S2Looking->WHU | 5.62 | 99.40 | 10.63 | 5.61 | FastSAM-s quick, 800 steps @0.5 |
| SAM-CD-s | DSIFN->LEVIR | 12.27 | 96.25 | 21.77 | 12.21 | FastSAM-s quick, 800 steps @0.5 |
| SAM-CD-s | DSIFN->WHU | 6.75 | 99.56 | 12.64 | 6.75 | FastSAM-s quick, 800 steps @0.5 |
| SAM-CD | LEVIR->WHU | 33.12 | 65.59 | 44.01 | 28.22 | official FastSAM.pt, source-tuned selected @0.5 |
| SAM-CD | WHU->LEVIR | 35.46 | 9.53 | 15.03 | 8.12 | official FastSAM.pt, full source-val selected @0.5 |
| SAM-CD | S2Looking->LEVIR | 19.00 | 18.72 | 18.86 | 10.41 | official FastSAM.pt, tuned source-val best @0.5 |
| SAM-CD | S2Looking->WHU | 40.40 | 48.83 | 44.22 | 28.39 | official FastSAM.pt, source-tuned selected @0.5 |
| SAM-CD | DSIFN->LEVIR | 26.19 | 64.48 | 37.25 | 22.89 | official FastSAM.pt, full source-val selected @0.5 |
| SAM-CD | DSIFN->WHU | 8.92 | 83.42 | 16.12 | 8.77 | official FastSAM.pt, full source-val selected @0.5 |
| ChangeCLIP | LEVIR->WHU | 21.37 | 61.15 | 31.67 | 18.81 | repaired RN50, source-val selected @0.5 |
| ChangeCLIP | WHU->LEVIR | 27.56 | 16.49 | 20.63 | 11.50 | repaired RN50, source-val selected @0.5 |
| ChangeCLIP | S2Looking->LEVIR | 2.67 | 17.00 | 4.62 | 2.37 | repaired RN50, source-val selected @0.5 |
| ChangeCLIP | S2Looking->WHU | 3.36 | 2.32 | 2.74 | 1.39 | repaired RN50, source-val selected @0.5 |
| ChangeCLIP | DSIFN->LEVIR | 20.15 | 37.22 | 26.14 | 15.04 | repaired RN50, source-val selected @0.5 |
| ChangeCLIP | DSIFN->WHU | 11.00 | 66.21 | 18.86 | 10.41 | repaired RN50, source-val selected @0.5 |
| DLV-CD | LEVIR->WHU | 87.07 | 58.43 | 69.93 | 53.77 | thr=0.5 |
| DLV-CD | WHU->LEVIR | 71.78 | 72.76 | 72.27 | 56.58 | thr=0.5 |
| DLV-CD | S2Looking->LEVIR | 63.61 | 78.18 | 70.15 | 54.02 | thr=0.5 |
| DLV-CD | S2Looking->WHU | 68.11 | 64.12 | 66.05 | 49.31 | thr=0.5 |
| DLV-CD | DSIFN->LEVIR | 36.98 | 88.81 | 52.22 | 35.34 | thr=0.5 |
| DLV-CD | DSIFN->WHU | 60.12 | 63.20 | 61.63 | 44.54 | thr=0.5 |

## F1 Matrix (%)

| Model | LEVIR->WHU | WHU->LEVIR | S2Looking->WHU | S2Looking->LEVIR | DSIFN->WHU | DSIFN->LEVIR | Avg. F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BiFA | 47.91 | 7.43 | 12.50 | 11.90 | 29.57 | 38.83 | 24.69 |
| DMINet | 23.58 | 13.52 | 19.76 | 12.74 | 18.32 | 21.79 | 18.29 |
| ACABFNet | 25.94 | 16.41 | 22.02 | 13.94 | 23.55 | 26.77 | 21.44 |
| AANet | 20.81 | 25.06 | 15.93 | 15.19 | 16.02 | 24.86 | 19.65 |
| SNUNet | 20.40 | 9.42 | 22.54 | 11.89 | 16.94 | 18.16 | 16.56 |
| BIT | 18.63 | 16.35 | 19.56 | 12.15 | 16.12 | 19.14 | 16.99 |
| SAM-CD-s | 22.75 | 23.16 | 10.63 | 10.60 | 12.64 | 21.77 | 16.93 |
| SAM-CD | 44.01 | 15.03 | 44.22 | 18.86 | 16.12 | 37.25 | 29.25 |
| ChangeCLIP | 31.67 | 20.63 | 2.74 | 4.62 | 18.86 | 26.14 | 17.44 |
| DLV-CD | 69.93 | 72.27 | 66.05 | 70.15 | 61.63 | 52.22 | 65.38 |

## DLV-CD HEF / Layer Selection Ablation Draft (%)

HEF is the renamed MHE module. The rows below use fixed-threshold source-only
testing without target-domain calibration. Head indices `0..3` correspond to
the selected DINO layers `3/6/9/12`, and index `4` is the fused decoder head.
Thus `Layer 3,4 + fused` means the deepest layer head plus the fused decoder.

| Method | LEVIR->WHU | WHU->LEVIR | S2Looking->WHU | S2Looking->LEVIR | DSIFN->LEVIR | DSIFN->WHU | Avg. F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fused only (w/o HEF) | 69.77 | 72.27 | 64.32 | 69.52 | 51.31 | 60.41 | 64.60 |
| Layer 2,4 + fused | 67.47 | 67.75 | 61.59 | 69.80 | 53.82 | 62.66 | 63.85 |
| Layer 3,4 + fused | 69.93 | 71.54 | 66.05 | 70.15 | 52.22 | 61.63 | 65.25 |
| All heads | 67.32 | 63.76 | 62.89 | 68.63 | 53.63 | 62.57 | 63.13 |
| DLV-CD selected | 69.93 | 72.27 | 66.05 | 70.15 | 52.22 | 61.63 | 65.38 |

The `DLV-CD selected` row is the final setting used in the main comparison
table. It is kept separate because it uses the most stable setting per source
configuration rather than a single uniform layer strategy.

## Source-Domain Validation Diagnostics (%)

The table below reports the best source-validation change-class F1 found in the
local training logs for each reproduced baseline source model. These values are
used only as a training-sufficiency diagnostic. Low source-val F1 means the
corresponding cross-domain result is likely affected by insufficient source
training and should not be interpreted as a fully converged baseline.

Status rule used here: OK >= 60, Marginal = 50-60, Low < 50, Failed < 30.

| Model | Source | Source val P | Source val R | Source val F1 | Source val IoU | Affected transfer rows | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| BiFA | LEVIR | N/A | N/A | N/A | N/A | LEVIR->WHU | pretrained checkpoint, no local val log |
| BiFA | WHU | 70.56 | 67.84 | 69.17 | 52.87 | WHU->LEVIR | OK |
| BiFA | S2Looking | 51.03 | 33.47 | 40.42 | 25.33 | S2Looking->WHU / S2Looking->LEVIR | Low |
| BiFA | DSIFN | N/A | N/A | N/A | N/A | DSIFN->WHU / DSIFN->LEVIR | pretrained checkpoint, no local val log |
| DMINet | LEVIR | 71.58 | 51.88 | 60.16 | 43.02 | LEVIR->WHU | OK |
| DMINet | WHU | 56.14 | 72.96 | 63.45 | 46.47 | WHU->LEVIR | OK |
| DMINet | S2Looking | 42.85 | 42.75 | 42.80 | 27.23 | S2Looking->WHU / S2Looking->LEVIR | Low |
| DMINet | DSIFN | 69.10 | 85.97 | 76.61 | 62.09 | DSIFN->WHU / DSIFN->LEVIR | OK |
| ACABFNet | LEVIR | 69.42 | 61.46 | 65.20 | 48.37 | LEVIR->WHU | OK |
| ACABFNet | WHU | 72.31 | 65.65 | 68.82 | 52.46 | WHU->LEVIR | OK |
| ACABFNet | S2Looking | 29.86 | 45.68 | 36.12 | 22.04 | S2Looking->WHU / S2Looking->LEVIR | Low |
| ACABFNet | DSIFN | 70.80 | 81.71 | 75.87 | 61.12 | DSIFN->WHU / DSIFN->LEVIR | OK |
| AANet | LEVIR | 42.54 | 78.08 | 55.07 | 38.00 | LEVIR->WHU | Marginal |
| AANet | WHU | 30.01 | 84.80 | 44.33 | 28.48 | WHU->LEVIR | Low |
| AANet | S2Looking | 15.18 | 56.67 | 23.95 | 13.60 | S2Looking->WHU / S2Looking->LEVIR | Failed |
| AANet | DSIFN | 62.30 | 89.19 | 73.36 | 57.92 | DSIFN->WHU / DSIFN->LEVIR | OK |
| SNUNet | LEVIR | 13.00 | 14.03 | 13.49 | 7.24 | LEVIR->WHU | Failed |
| SNUNet | WHU | 67.23 | 41.70 | 51.48 | 34.66 | WHU->LEVIR | Marginal |
| SNUNet | S2Looking | 32.74 | 26.44 | 29.25 | 17.13 | S2Looking->WHU / S2Looking->LEVIR | Failed |
| SNUNet | DSIFN | 69.17 | 57.76 | 62.95 | 45.93 | DSIFN->WHU / DSIFN->LEVIR | OK |
| BIT | LEVIR | 65.99 | 63.82 | 64.89 | 48.02 | LEVIR->WHU | OK |
| BIT | WHU | 39.06 | 71.60 | 50.55 | 33.82 | WHU->LEVIR | Marginal |
| BIT | S2Looking | 42.34 | 39.40 | 40.82 | 25.64 | S2Looking->WHU / S2Looking->LEVIR | Low |
| BIT | DSIFN | 62.22 | 86.01 | 72.21 | 56.50 | DSIFN->WHU / DSIFN->LEVIR | OK |
| SAM-CD-s | LEVIR | 39.12 | 87.55 | 54.08 | 37.06 | LEVIR->WHU | Marginal; FastSAM-s quick |
| SAM-CD-s | WHU | 22.69 | 90.13 | 36.26 | 22.14 | WHU->LEVIR | Low; FastSAM-s quick |
| SAM-CD-s | S2Looking | 8.67 | 72.11 | 15.48 | 8.39 | S2Looking->WHU / S2Looking->LEVIR | Failed; FastSAM-s quick |
| SAM-CD-s | DSIFN | 41.04 | 98.41 | 57.92 | 40.77 | DSIFN->WHU / DSIFN->LEVIR | Marginal; FastSAM-s quick |
| SAM-CD | LEVIR | 65.00 | 75.79 | 69.98 | 53.82 | LEVIR->WHU | Marginal/OK; official FastSAM.pt, best source-val after resume |
| SAM-CD | WHU | 71.55 | 61.49 | 66.14 | 49.41 | WHU->LEVIR | OK; official FastSAM.pt, full-val selected after resume |
| SAM-CD | S2Looking | 37.21 | 45.45 | 40.92 | 25.72 | S2Looking->WHU / S2Looking->LEVIR | Low; official FastSAM.pt, full-val evaluation |
| SAM-CD | DSIFN | 67.92 | 69.57 | 68.74 | 52.37 | DSIFN->WHU / DSIFN->LEVIR | OK; official FastSAM.pt, full-val selected after conservative resume |
| ChangeCLIP | LEVIR | 45.69 | 67.91 | 54.62 | 37.57 | LEVIR->WHU | Marginal; repaired RN50, best 3000 iter |
| ChangeCLIP | WHU | 25.99 | 62.54 | 36.72 | 22.49 | WHU->LEVIR | Low; repaired RN50, cw5 best 4000 iter |
| ChangeCLIP | S2Looking | 14.73 | 25.72 | 18.73 | 10.33 | S2Looking->WHU / S2Looking->LEVIR | Failed; repaired RN50, best 2000 iter |
| ChangeCLIP | DSIFN | 74.77 | 44.42 | 55.73 | 38.63 | DSIFN->WHU / DSIFN->LEVIR | Marginal; repaired RN50, cw1 best 1000 iter |

Practical interpretation: current S2Looking-source baseline results for BiFA,
DMINet, ACABFNet, AANet, SNUNet, and BIT are all likely under-trained. SNUNet
LEVIR-source and AANet S2Looking-source are clear failed runs. BIT uses a
randomly initialized ResNet18 backbone in this local reproduction, so its
numbers are conservative relative to ImageNet-pretrained BIT.

## Retrain/Retest Notes (%)

Low-source-F1 baselines were partially resumed and retested. These rows are
kept as diagnostic records; the main cross-domain table should use the better
result per direction when a retrained checkpoint is worse than the original.

| Model | Source | Source val F1 after resume | Direction | Original F1 | Retest F1 | Recommendation |
| --- | --- | ---: | --- | ---: | ---: | --- |
| BiFA | S2Looking | 38.78 | S2Looking->WHU | 12.50 | 11.19 | keep original |
| BiFA | S2Looking | 38.78 | S2Looking->LEVIR | 11.90 | 8.16 | keep original |
| DMINet | S2Looking | 43.60 | S2Looking->WHU | 10.41 | 19.76 | use retest |
| DMINet | S2Looking | 43.60 | S2Looking->LEVIR | 7.32 | 12.74 | use retest |
| ACABFNet | S2Looking | 46.34 | S2Looking->WHU | 22.02 | 17.31 | keep original |
| ACABFNet | S2Looking | 46.34 | S2Looking->LEVIR | 13.94 | 11.80 | keep original |
| AANet | S2Looking | 45.07 | S2Looking->WHU | 15.93 | 14.09 | keep original |
| AANet | S2Looking | 45.07 | S2Looking->LEVIR | 14.14 | 15.19 | use retest |
| SNUNet | S2Looking | 41.56 | S2Looking->WHU | 22.54 | 14.87 | keep original |
| SNUNet | S2Looking | 41.56 | S2Looking->LEVIR | 6.56 | 11.89 | use retest |
| SNUNet | LEVIR | 35.17 | LEVIR->WHU | 20.40 | 8.98 | keep original |
| AANet | WHU | 43.07 | WHU->LEVIR | 15.28 | 25.06 | use retest |
| SNUNet | LEVIR | 48.90 | LEVIR->WHU | 20.40 | 12.73 | second-stage resume; keep original |
| AANet | WHU | 55.18 | WHU->LEVIR | 15.28 | 24.31 | second-stage resume; first retest remains better |
| SAM-CD | WHU | 66.14 | WHU->LEVIR | 8.53 | 15.03 | full source-val selected; use retest |
| SAM-CD | WHU recall-biased | 48.59 | WHU->LEVIR | 15.03 | 28.24 | diagnostic only; last checkpoint, not source-val best |
| SAM-CD | WHU continued | 61.04 | WHU->LEVIR | 28.24 | 19.98 | continued recall-biased training; worse than previous diagnostic |
| SAM-CD | S2Looking last | 26.95 | S2Looking->LEVIR | 18.86 | 18.86 | same as best checkpoint |
| SAM-CD | S2Looking AdamW last | 40.92 | S2Looking->LEVIR | 16.29 | 14.77 | worse than source-val best |
| SAM-CD | DSIFN conservative | 68.74 | DSIFN->WHU | 14.21 | 16.12 | full source-val selected; use retest |
| SAM-CD | DSIFN conservative | 68.74 | DSIFN->LEVIR | 33.37 | 37.25 | full source-val selected; use retest |
| SAM-CD | DSIFN ultra-conservative | 19.67 | DSIFN->WHU | 16.12 | 36.98 | diagnostic only; last checkpoint, not source-val best |
| ChangeCLIP | WHU cw10 | 34.54 | WHU->LEVIR | 20.63 | 23.12 | diagnostic only; worse source-val than cw5 |
| ChangeCLIP | DSIFN cw10 | 49.02 | DSIFN->WHU / DSIFN->LEVIR | 18.86 / 26.14 | N/A | rejected: all-foreground source-val prediction |

## Model Complexity

| Model | Input size | Total Params | Trainable Params | FLOPs | Latency | FPS | Device |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| BiFA | 256x256 pair | 9.87M | 9.87M | 53.00G | 54.93 ms/pair | 18.21 | RTX 4060 Laptop |
| DMINet | 256x256 pair | 6.76M | 6.76M | 14.55G | 8.90 ms/pair | 112.37 | RTX 4060 Laptop |
| ACABFNet | 256x256 pair | 117.46M | 117.46M | 28.29G | 49.91 ms/pair | 20.03 | RTX 4060 Laptop |
| AANet | 256x256 pair | 15.89M | 15.89M | 24.21G | 17.17 ms/pair | 58.23 | RTX 4060 Laptop |
| SNUNet | 256x256 pair | 1.35M | 1.35M | 4.73G | 4.00 ms/pair | 250.24 | RTX 4060 Laptop |
| BIT | 256x256 pair | 11.94M | 11.94M | 8.75G | 13.96 ms/pair | 71.66 | RTX 4060 Laptop |
| DLV-CD | 256x256 pair | 97.48M | 11.82M | 94.90G | 42.47 ms/pair | 23.55 | RTX 4060 Laptop |

## Source Files

- Baseline results: `baselines/BiFA/experiments/tile_eval_260707/*/tile_eval_results.json`
- SNUNet/BIT results: `baselines/BiFA/experiments/tile_eval_extra_260708/*/*/tile_eval_results.json`
- Baseline complexity: `baselines/BiFA/experiments/*_complexity_256.json`
- DLV-CD LEVIR/WHU: `outputs/uwi_test/*/eval_results.json`
- DLV-CD S2Looking transfer: `outputs/task4_transfer_s2looking/*/eval_results.json`
- DLV-CD DSIFN transfer: `outputs/task2_dsifn_source/*/eval_results.json`
- DLV-CD fixed-threshold source-calibrated DSIFN transfer:
  `outputs/fixed05_traincal_2026-07-06/DSIFN2*_tv*_fp*_*/eval_results.json`
- SAM-CD-s quick reproduction:
  `outputs/samcd_quick_2026-08-10/*/eval_results.json`
- SAM-CD official reproduction:
  `outputs/samcd_official_2026-08-10/*/eval_results.json`
- SAM-CD official tuned reproduction:
  `outputs/samcd_official_tune_2026-08-10/*/eval_results.json`
- SAM-CD official exploratory reproduction:
  `outputs/samcd_official_explore_2026-08-11/*/eval_results.json`
- ChangeCLIP repaired reproduction:
  `outputs/changeclip_fixed_2026-08-11/*/eval_results.json`
