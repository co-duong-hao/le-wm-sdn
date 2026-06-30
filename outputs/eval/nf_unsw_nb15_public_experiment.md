# NF-UNSW-NB15 Public Experiment Report

Date: 2026-06-29

## Dataset

The public dataset used in this run is NF-UNSW-NB15 from the University of
Queensland NetFlow NIDS dataset collection. The downloaded RDM archive contains:

```text
NF-UNSW-NB15.csv
```

The RDM API endpoint used for the public download metadata was:

```text
https://api.rdm.uq.edu.au/datasets/c31a9f50-ef99-11ed-ab7b-c7846b13c8a9/download
```

The ZIP was saved to:

```text
data/raw/nf_unsw_nb15/nf_unsw_nb15_rdm.zip
```

## Conversion

Command:

```powershell
python scripts/prepare_flow_csv_graph.py `
  --input data\raw\nf_unsw_nb15\nf_unsw_nb15_rdm.zip `
  --output data\processed\nf_unsw_nb15_graph.npz `
  --extract-dir data\raw\nf_unsw_nb15\extracted `
  --top-nodes 64 `
  --rows-per-bin 1000
```

Observed output:

```text
saved=data\processed\nf_unsw_nb15_graph.npz
steps=1624 nodes=44
edges=294 attack_steps=1084
```

Converted tensor summary:

```text
node_features: (1624, 44, 8)
edge_index:    (2, 294)
edge_features: (1624, 294, 5)
label:         (1624,)
normal_steps:  540
attack_steps:  1084
```

## Protocol

The source CSV is sorted with attack-heavy graph windows first and normal-heavy
windows later. A prefix split would be misleading. The experiment therefore
uses a stratified 50/50 train/test split:

```powershell
python scripts/run_baseline_comparison.py `
  --data data\processed\nf_unsw_nb15_graph.npz `
  --output outputs\eval\nf_unsw_nb15_baseline_comparison.json `
  --history 8 `
  --latent-dim 16 `
  --split stratified `
  --eval-split test `
  --test-fraction 0.5 `
  --percentile 99 `
  --seed 3072
```

## Results

| Method | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|
| One-class feature distance | 0.8421 | 0.0595 | 0.1111 | 0.3663 |
| Supervised logistic regression | 0.9818 | 1.0000 | 0.9908 | 0.9876 |
| Supervised NumPy MLP | 0.9818 | 1.0000 | 0.9908 | 0.9876 |
| Supervised gradient-boosted stumps | 0.9809 | 0.9554 | 0.9680 | 0.9579 |
| Supervised PyTorch MLP | 0.9779 | 0.9888 | 0.9834 | 0.9777 |
| Supervised PyTorch CNN | 0.9781 | 0.9963 | 0.9871 | 0.9827 |
| Supervised PyTorch LSTM | 0.9799 | 0.9963 | 0.9880 | 0.9839 |
| Raw-feature ridge surprise | 0.8000 | 0.0669 | 0.1235 | 0.3676 |
| LeWM-SDN latent surprise + phase | 0.7273 | 0.0446 | 0.0841 | 0.3527 |
| PyTorch Temporal GNN JEPA, calibrated | 0.7731 | 0.9994 | 0.8718 | 0.8037 |

## PyTorch Temporal GNN + SDN-JEPA Public Run

Training command:

```powershell
.venv\Scripts\python.exe train.py `
  data.path=data\processed\nf_unsw_nb15_graph.npz `
  data.num_steps=10 history_size=4 embed_dim=32 prog_dim=8 `
  model.encoder.hidden_dim=32 model.encoder.depth=2 `
  model.predictor.depth=1 model.predictor.heads=4 `
  model.predictor.mlp_dim=128 model.predictor.dim_head=8 `
  trainer.max_epochs=100 loader.batch_size=128 loader.pin_memory=false `
  optimizer.lr=0.0002 optimizer.weight_decay=0.0005 `
  output_dir=outputs\checkpoints\nf_unsw_torch_100e `
  output_model_name=nf_unsw_torch_100e checkpoint_interval=25 `
  loss.sigreg.kwargs.num_proj=16
```

Training trace:

```text
epoch=1 train_loss=8.306644 val_loss=3.717935
epoch=20 train_loss=5.437995 val_loss=2.588119
epoch=40 train_loss=4.348165 val_loss=1.935771
epoch=60 train_loss=3.635736 val_loss=1.439211
epoch=80 train_loss=3.035367 val_loss=1.144290
epoch=100 train_loss=2.406026 val_loss=1.064960
```

Evaluation output:

```text
outputs/eval/nf_unsw_torch_sdn_jepa_eval.json
```

Default 99th-percentile normal-threshold metrics:

```text
precision=0.0577 recall=0.0003 f1=0.0006 accuracy=0.3287
```

Score diagnostics:

```text
normal_score_mean=0.3271
attack_score_mean=0.4622
test_auc_high_score_is_attack=0.7199
```

Threshold calibration was performed on a held-out calibration split using
normal-score quantiles, normal mean plus k standard deviations, and best-F1
selection on calibration labels. The selected threshold was then evaluated on
the held-out test split:

```text
strategy=best_f1_on_calibration_high_tail
threshold=0.2654024804
precision=0.7731 recall=0.9994 f1=0.8718 accuracy=0.8037
balanced_accuracy=0.7045
```

A 200-epoch CPU run was also tested with the same compact architecture, but its
selected calibrated held-out F1 was lower at 0.8311. The 100-epoch checkpoint
is therefore used for the reported public neural result.

## Interpretation

This run addresses the previous "synthetic-only" weakness by adding a public
UNSW-derived benchmark. The result is not favorable to the current anomaly
formulation: supervised baselines perform well, while one-class and predictive
surprise baselines have low recall.

The correct paper claim is therefore limited:

- The public-data conversion and evaluation path now works.
- Supervised baselines are strong on NF-UNSW-NB15.
- The current NumPy LeWM-SDN reference score is not competitive on this public
  split.
- The PyTorch Temporal GNN + SDN-JEPA trains and evaluates on public data.
- Default 99th-percentile thresholding is still poor, but calibration changes
  the public neural result from near-zero F1 to F1 = 0.8718.
- The calibrated public neural result is useful, but it still trails supervised
  baselines and needs SDN-native validation on InSDN.
