# LeWM-SDN Offline Experiment Report

Date: 2026-06-29

## Purpose

This report records executable experiments for the current LeWM-SDN project.
Because the machine cannot reach the public InSDN download server, the
experiment uses an offline synthetic SDN/DDoS graph dataset. The report records
both a NumPy reference implementation of the LeWM-SDN anomaly-scoring idea and
an executed PyTorch Temporal GNN + SDN-JEPA smoke run.

The experiment is intended to prove that the project data path, graph format,
latent prediction workflow, anomaly scoring, and reporting pipeline can run
successfully in the current environment.

## Dataset Choice

The selected dataset for this executable run is:

```text
data/processed/synthetic_sdn_ddos.npz
```

This dataset is generated locally by `scripts/generate_synthetic_sdn.py`.
It is appropriate for the current offline stage because it has the same tensor
interface expected by the SDN-JEPA code:

```text
node_features: dynamic SDN node metrics
edge_index:    static directed network topology
edge_features: dynamic link/flow metrics
label:         normal/attack state per timestep
action:        no-op mitigation action id
```

It is not a substitute for a final external benchmark such as InSDN or
UNSW-NB15. It is a controlled offline dataset for proving that the current
project is runnable.

## Commands Executed

Generate the dataset:

```powershell
& "C:\Users\ADMIN\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" `
  scripts\generate_synthetic_sdn.py `
  --output data\processed\synthetic_sdn_ddos.npz `
  --steps 1200 `
  --seed 3072
```

Observed output:

```text
saved=data\processed\synthetic_sdn_ddos.npz
steps=1200 nodes=29 edges=64
attack_steps=240 normal_steps=960
```

Run the NumPy SDN-JEPA reference experiment:

```powershell
& "C:\Users\ADMIN\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" `
  scripts\run_numpy_sdn_jepa_experiment.py `
  --data data\processed\synthetic_sdn_ddos.npz `
  --output outputs\eval\numpy_sdn_jepa_experiment.json `
  --history 8 `
  --latent-dim 16 `
  --ridge 1 `
  --alpha 1 `
  --beta 0.2
```

Run the baseline comparison requested by expert feedback:

```powershell
& "C:\Users\ADMIN\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" `
  scripts\run_baseline_comparison.py `
  --data data\processed\synthetic_sdn_ddos.npz `
  --output outputs\eval\baseline_comparison.json `
  --history 8 `
  --latent-dim 16 `
  --warmup 600
```

Run supervised PyTorch neural baselines requested by expert feedback:

```powershell
.venv\Scripts\python.exe `
  scripts\run_torch_supervised_baselines.py `
  --data data\processed\synthetic_sdn_ddos.npz `
  --output outputs\eval\torch_supervised_baselines.json `
  --history 8 `
  --warmup 600 `
  --hidden-dim 64 `
  --epochs 80
```

Run the actual PyTorch Temporal GNN + SDN-JEPA smoke training:

```powershell
.venv\Scripts\python.exe train.py `
  data.path=data\processed\synthetic_sdn_ddos.npz `
  data.num_steps=10 history_size=4 embed_dim=32 prog_dim=8 `
  model.encoder.hidden_dim=32 model.encoder.depth=2 `
  model.predictor.depth=1 model.predictor.heads=4 `
  model.predictor.mlp_dim=128 model.predictor.dim_head=8 `
  trainer.max_epochs=8 loader.batch_size=64 `
  output_dir=outputs\checkpoints\synthetic_torch `
  output_model_name=synthetic_torch loss.sigreg.kwargs.num_proj=16
```

Observed training trace:

```text
epoch=1 train_loss=4.804901 val_loss=3.797070
epoch=2 train_loss=4.474140 val_loss=3.594635
epoch=3 train_loss=4.406020 val_loss=4.089411
epoch=4 train_loss=4.159748 val_loss=3.774366
epoch=5 train_loss=3.953923 val_loss=3.237516
epoch=6 train_loss=3.775210 val_loss=3.281291
epoch=7 train_loss=3.718038 val_loss=3.441900
epoch=8 train_loss=3.715788 val_loss=3.177553
```

Run trained-model evaluation:

```powershell
.venv\Scripts\python.exe eval.py `
  checkpoint=outputs\checkpoints\synthetic_torch\synthetic_torch_best.pt `
  data.path=data\processed\synthetic_sdn_ddos.npz `
  data.num_steps=10 `
  output.path=outputs\eval\torch_sdn_jepa_eval.json
```

## Method

The reference experiment mirrors the core LeWM-SDN idea without PyTorch:

1. Aggregate dynamic graph snapshots into graph-level state vectors.
2. Learn a latent representation from normal states using PCA.
3. Treat the first two latent dimensions as `z_prog` and compute phase:

```text
theta_t = atan2(z_prog[1], z_prog[0])
```

4. Train a ridge next-state predictor on normal windows.
5. Score each transition with:

```text
A_t = alpha * prediction_surprise + beta * phase_drift
```

This is not a replacement for `train.py`; it is a runnable reference experiment
for the same predictive-latent anomaly-detection principle.

## Results

Result file:

```text
outputs/eval/numpy_sdn_jepa_experiment.json
```

Key output:

```json
{
  "method": "numpy_sdn_jepa_reference",
  "num_transitions": 1192,
  "threshold": 5.005514621734619,
  "normal_score_mean": 1.0056030750274658,
  "attack_score_mean": 1209.7738037109375,
  "surprise_mean": 244.2268829345703,
  "phase_drift_mean": 0.7744392156600952,
  "metrics": {
    "tp": 240,
    "tn": 942,
    "fp": 10,
    "fn": 0,
    "precision": 0.96,
    "recall": 1.0,
    "f1": 0.9795918367346939,
    "accuracy": 0.9916107382550335
  }
}
```

Baseline comparison output:

```text
outputs/eval/baseline_comparison.json
outputs/eval/torch_supervised_baselines.json
outputs/eval/torch_sdn_jepa_eval.json
```

| Method | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|
| One-class feature distance | 0.9449 | 1.0000 | 0.9717 | 0.9883 |
| Supervised logistic regression | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Supervised NumPy MLP | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Supervised gradient-boosted stumps | 0.9796 | 1.0000 | 0.9897 | 0.9958 |
| Supervised PyTorch MLP | 0.9875 | 0.9875 | 0.9875 | 0.9950 |
| Supervised PyTorch CNN | 0.9833 | 0.9833 | 0.9833 | 0.9933 |
| Supervised PyTorch LSTM | 0.9833 | 0.9833 | 0.9833 | 0.9933 |
| Raw-feature ridge surprise | 0.7091 | 0.1625 | 0.2644 | 0.8180 |
| LeWM-SDN latent surprise + phase | 0.9057 | 1.0000 | 0.9505 | 0.9790 |
| PyTorch Temporal GNN JEPA, 8 epochs | 0.0753 | 0.0032 | 0.0062 | 0.7911 |

## Interpretation

The experiment succeeds under the current offline constraints:

- The SDN graph dataset is generated correctly.
- The project can load graph tensors in the target shape.
- Normal latent dynamics can be learned from normal windows.
- DDoS windows produce much higher anomaly scores than normal windows.
- The fairer train-normal threshold baseline reaches F1 = 0.9505 on the
  generated dataset; the earlier standalone NumPy reference reaches F1 = 0.9796.

This supports the project direction: DDoS can be detected as a violation of
predicted latent network dynamics rather than by static flow classification.
However, the supervised baselines solve the generated dataset nearly perfectly,
including the lightweight boosting baseline, so the synthetic dataset is not
strong enough for final benchmark claims.
The PyTorch neural model now trains and evaluates, but its anomaly score is
weak in the short CPU smoke run. This addresses the previous "not trained"
defect as an engineering milestone, not as a strong detection result.

The public NF-UNSW-NB15 benchmark path has also been executed:

```text
data/processed/nf_unsw_nb15_graph.npz
outputs/eval/nf_unsw_nb15_baseline_comparison.json
outputs/eval/nf_unsw_nb15_public_experiment.md
```

On the public held-out split, supervised logistic regression and NumPy MLP
reach F1 = 0.9908, while the current LeWM-SDN latent surprise + phase reference
reaches only F1 = 0.0841. This resolves the synthetic-only defect but shows
that the current anomaly score is not competitive on the public benchmark.
The public PyTorch Temporal GNN + SDN-JEPA checkpoint also trains and evaluates,
but its default threshold F1 is 0.0000.

## Remaining Work

For final scientific reporting, extend the public benchmark work:

1. Tune and calibrate the PyTorch Temporal GNN + SDN-JEPA on
   `nf_unsw_nb15_graph.npz`.
2. Add InSDN, if the UCD/ASEADOS server becomes reachable.
3. Add UNSW-NB15 train/test CSVs if they can be copied offline into
   `data/raw/unsw_nb15`.
4. Add CICIDS2017 or CICDDoS2019 for DDoS-focused external validation.

Public dataset availability checked on 2026-06-29:

- The official UNSW-NB15 page states that the dataset contains four full CSV
  files and partitioned training/testing CSV files named
  `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv`, with 175,341
  training records and 82,332 testing records.
- The official download link redirects to UNSW SharePoint / Microsoft login in
  this environment, so the files could not be downloaded automatically.
- The repository therefore includes `scripts/prepare_unsw_nb15.py`; once those
  two CSV files are copied into `data/raw/unsw_nb15`, the public benchmark can
  be converted and evaluated.

For a larger PyTorch SDN-JEPA experiment, use the local `.venv` or another
environment with PyTorch installed and run:

```bash
.venv/Scripts/python.exe train.py data.path=data/processed/synthetic_sdn_ddos.npz data.num_steps=16
.venv/Scripts/python.exe eval.py checkpoint=outputs/checkpoints/lewm_sdn/lewm_sdn_best.pt data.path=data/processed/synthetic_sdn_ddos.npz data.num_steps=16
```

The current paper draft now uses Springer LNCS format and includes figures,
baseline comparison, method equations, SIGReg details, and MPC/CEM planning
formulas. The PyTorch training path has been executed on synthetic data. The
public-dataset concern is resolved at benchmark-path level by NF-UNSW-NB15;
public-data PyTorch JEPA tuning remains open. LNCS page-count verification is
complete: MiKTeX `pdflatex` renders `main.pdf` as 16 pages.
