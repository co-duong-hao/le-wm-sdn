# Expert Feedback Audit

Date: 2026-06-29

This file tracks the nine expert-review issues against the current repository
state. It is intentionally conservative: an item is marked complete only when
there is executable or file-level evidence in the workspace.

## Status Summary

| # | Issue | Current status | Evidence |
|---:|---|---|---|
| 1 | No baseline comparison | Addressed for synthetic and public NF-UNSW-NB15 baselines | `outputs/eval/baseline_comparison.json`, `outputs/eval/nf_unsw_nb15_baseline_comparison.json`, `outputs/eval/torch_supervised_baselines.json`, `main.tex` Tables `tab:baseline` and `tab:publicbaseline` |
| 2 | Dataset is synthetic and too small | Partially addressed | Public NF-UNSW-NB15 was converted locally and the metrics are recorded in `outputs/eval/nf_unsw_nb15_baseline_comparison.json`; PyTorch JEPA has not yet been tuned on it |
| 3 | NumPy reference is not trained neural JEPA | Addressed as an executable path, not as a strong result | PyTorch Temporal GNN + SDN-JEPA runs were executed for synthetic and public NF-UNSW-NB15; public default-threshold F1 is 0.0 |
| 4 | Paper length below 12-20 pages | Addressed | MiKTeX `pdflatex` renders `main.pdf` as 16 pages; see `outputs/eval/paper_build_report.md` |
| 5 | Wrong format, not LNCS | Addressed | `main.tex` uses `\documentclass[runningheads]{llncs}` and `llncs.cls` is present |
| 6 | Related Work too thin | Addressed in draft | `main.tex` has a Related Work section and about 20-30 bibliography entries |
| 7 | Method lacks detail | Addressed in draft | `main.tex` includes Temporal GNN equations, latent split, SIGReg, anomaly score, and moves unevaluated MPC/CEM mitigation to Future Work |
| 8 | Contributions mismatch experiments | Improved, still limited | Contributions now state offline/public validation and caveats; public PyTorch JEPA is executed but not competitive |
| 9 | No figures/diagrams | Addressed | `figures/architecture_pipeline.png`, `dataset_timeline.png`, `baseline_f1_comparison.png`, `score_separation.png` |

## Baseline Evidence

The current baseline suite includes:

- one-class feature distance
- supervised logistic regression
- supervised NumPy MLP
- supervised gradient-boosted decision stumps
- supervised PyTorch MLP
- supervised PyTorch CNN
- supervised PyTorch LSTM
- raw-feature ridge next-state surprise
- LeWM-SDN latent surprise + phase reference
- PyTorch Temporal GNN + SDN-JEPA implementation sanity check

Key recorded metrics on `data/processed/synthetic_sdn_ddos.npz`:

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
| PyTorch JEPA sanity check | 0.0753 | 0.0032 | 0.0062 | 0.7911 |

Interpretation: this fixes the absence of baseline comparison, but it also
shows that the synthetic dataset is too easy for supervised baselines. These
results should not be presented as final benchmark evidence.

## Public Dataset Evidence

The project records a public UNSW-derived NetFlow benchmark result. The raw ZIP
and generated graph tensor are local reproducibility artifacts and are not
tracked in the conference repo.

```text
outputs/eval/nf_unsw_nb15_baseline_comparison.json
outputs/eval/nf_unsw_nb15_public_experiment.md
```

Converted graph summary:

```text
steps=1624 nodes=44 edges=294
normal_steps=540 attack_steps=1084
```

Public NF-UNSW-NB15 held-out test metrics:

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
| PyTorch JEPA sanity check | 0.0000 | 0.0000 | 0.0000 | 0.3285 |

## Public Dataset Status

Local files used during the executed run:

```text
data/processed/synthetic_sdn_ddos.npz
data/raw/nf_unsw_nb15/nf_unsw_nb15_rdm.zip
data/processed/nf_unsw_nb15_graph.npz
data/raw/insdn/
```

The project includes one recorded public benchmark result and retains converters
for reproducing or extending the run:

- `scripts/prepare_insdn.py`
- `scripts/prepare_unsw_nb15.py`
- `scripts/prepare_flow_csv_graph.py`

Expected next commands after copying InSDN:

```bash
python scripts/prepare_insdn.py \
  --input data/raw/insdn/InSDN_DatasetCSV.zip \
  --output data/processed/insdn_graph.npz

python train.py data.path=data/processed/insdn_graph.npz data.num_steps=16
python eval.py checkpoint=outputs/checkpoints/lewm_sdn/lewm_sdn_best.pt \
  data.path=data/processed/insdn_graph.npz data.num_steps=16
```

## Neural JEPA Status

The proposed Temporal GNN + SDN-JEPA path is implemented and has been executed
on both synthetic and public NF-UNSW-NB15 graphs. Checkpoint weights are
generated artifacts and are not tracked in Git; the recorded evaluation files
are:

```text
outputs/eval/torch_sdn_jepa_eval.json
outputs/eval/nf_unsw_torch_sdn_jepa_eval.json
```

Training loss decreased in the recorded CPU sanity checks, but anomaly
detection quality is poor. This proves the neural code path runs; it does not
prove the model is competitive.

## Formatting and Paper Status

The paper now uses Springer LNCS:

```tex
\documentclass[runningheads]{llncs}
```

The workspace contains `llncs.cls`. MiKTeX `pdflatex` renders the paper
successfully:

```text
main.pdf: 16 pages
```

This satisfies the 12-20 page requirement.

## Remaining Blocking Requirement

The synthetic-only blocker is resolved by the NF-UNSW-NB15 public benchmark.
The remaining scientific blockers are:

- PyTorch Temporal GNN + SDN-JEPA has been trained/evaluated on public
  NF-UNSW-NB15, but it is not competitive and needs tuning/calibration.
- Remaining work is now quality/tuning work rather than a missing format or
  missing execution path.
