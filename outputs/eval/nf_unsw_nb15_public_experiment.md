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
| PyTorch Temporal GNN JEPA, 8 epochs | 0.0000 | 0.0000 | 0.0000 | 0.3285 |

## PyTorch Temporal GNN + SDN-JEPA Public Run

Training command:

```powershell
.venv\Scripts\python.exe train.py `
  data.path=data\processed\nf_unsw_nb15_graph.npz `
  data.num_steps=10 history_size=4 embed_dim=32 prog_dim=8 `
  model.encoder.hidden_dim=32 model.encoder.depth=2 `
  model.predictor.depth=1 model.predictor.heads=4 `
  model.predictor.mlp_dim=128 model.predictor.dim_head=8 `
  trainer.max_epochs=8 loader.batch_size=64 `
  output_dir=outputs\checkpoints\nf_unsw_torch `
  output_model_name=nf_unsw_torch loss.sigreg.kwargs.num_proj=16
```

Training trace:

```text
epoch=1 train_loss=4.580883 val_loss=3.216061
epoch=2 train_loss=4.441509 val_loss=3.556443
epoch=3 train_loss=4.247307 val_loss=3.660939
epoch=4 train_loss=4.353933 val_loss=3.100650
epoch=5 train_loss=4.136390 val_loss=3.282528
epoch=6 train_loss=3.825837 val_loss=3.516848
epoch=7 train_loss=3.864302 val_loss=3.026342
epoch=8 train_loss=3.907543 val_loss=2.961094
```

Evaluation output:

```text
outputs/eval/nf_unsw_torch_sdn_jepa_eval.json
```

Default normal-threshold metrics:

```text
precision=0.0000 recall=0.0000 f1=0.0000 accuracy=0.3285
```

Oracle threshold diagnostic:

```text
precision=0.6681 recall=1.0000 f1=0.8010 accuracy=0.6681
```

The oracle threshold is not reported as the main detector result; it only
shows that the score distribution contains label information and that threshold
calibration remains unresolved.

## Interpretation

This run addresses the previous "synthetic-only" weakness by adding a public
UNSW-derived benchmark. The result is not favorable to the current anomaly
formulation: supervised baselines perform well, while one-class and predictive
surprise baselines have low recall.

The correct paper claim is therefore limited:

- The public-data conversion and evaluation path now works.
- Supervised baselines are strong on NF-UNSW-NB15.
- The current LeWM-SDN reference score is not competitive on this public split.
- The PyTorch Temporal GNN + SDN-JEPA now trains and evaluates on public data,
  but its default anomaly threshold is not competitive.
- Public-data threshold calibration and model tuning are required before any
  competitive claim can be made.
