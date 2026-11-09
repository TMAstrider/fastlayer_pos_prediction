# Public pyramid orientation model comparison

## Setup

All three experiments use the same 10,000-sample public pyramid dataset and official split (8,100 train / 900 validation / 1,000 held-out test). Each sample is represented by 2,048 surface XYZ points centered and scaled to the unit sphere. The target is the metadata build-up axis, and all models use normalized 3-D outputs with cosine direction loss `1 - dot(prediction, target)`. Yaw is not predicted.

PTv3 uses the Utonia Tiny encoder in encoder-only mode, multi-stage mean/max pooling (972-D feature), and a 972→256→64→3 direction head. The pretrained run fine-tunes the full encoder and head with separate learning rates. PTv3 uses a 64-point attention patch and batches of 32 with 2-step gradient accumulation (effective batch 64).

## Held-out test results

| Model | First validation epoch <3° | Best epoch | Validation mean | Test mean | Test median | Test P90 | ≤1° | ≤3° | ≤5° | Train time | Peak GPU memory |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PointNet（从头训练） | 10 | 157 | 0.71° | 0.71° | 0.65° | 1.25° | 78.0% | 99.9% | 100.0% | 20.8 min | 2.47 GiB |
| PTv3（从头训练） | 6 | 12 | 1.96° | 2.00° | 1.67° | 3.59° | 22.9% | 82.8% | 97.3% | 21.2 min | 1.84 GiB |
| PTv3（预训练微调） | 6 | 12 | 2.10° | 2.19° | 1.95° | 3.81° | 15.1% | 79.5% | 96.3% | 21.4 min | 1.87 GiB |

Validation curves: [`validation_curves.png`](../runs/public_pyramids_model_comparison/final/validation_curves.png).

## Interpretation

On this controlled geometry task, the lowest test angular error came from **PointNet（从头训练）**. All three validation curves first went below 3°: PointNet at epoch 10, scratch PTv3 at epoch 6, and pretrained PTv3 at epoch 6. Scratch PTv3 briefly rose above 3° at epoch 7, then remained below from epoch 8 onward. PTv3 continued improving gradually through epoch 12, while PointNet continued to improve with its longer run. This supports using 3° as a stopping target for future runs when that precision is sufficient, with a two-epoch confirmation to avoid stopping on a one-epoch dip.

This run evaluates one target axis against the dataset's single reference axis. It does not assess equivalent/symmetric orientations, support amount, manufacturability constraints, yaw, or transfer to the 193 real parts. PointNet ran for 160 epochs and PTv3 for 12, so use these results as an initial comparison; a validation-converged follow-up on the real-part split is needed before drawing strong architectural conclusions.

## Artifacts

- Checkpoints and per-run metrics: `runs/public_pyramids_model_comparison/final/`.
- Test predictions: one `test_predictions.npz` per run.
- Combined metrics: `runs/public_pyramids_model_comparison/final/summary.json`.
- Training script: `tools/train_public_model_comparison.py`.
