# Offline Synthetic SDN/DDoS Experiment

Date: 2026-06-29

## Reason

The InSDN website could not be reached from the current environment, so an
offline SDN/DDoS dataset was generated to validate the project data path.

## Dataset

Generated file:

```text
data/processed/synthetic_sdn_ddos.npz
```

Shape summary:

```text
node_features: (1200, 29, 8)
edge_index:    (2, 64)
edge_features: (1200, 64, 5)
label:         (1200,)
action:        (1200,)
```

Label summary:

```text
normal_steps: 960
attack_steps: 240
```

## Command

```bash
python scripts/generate_synthetic_sdn.py \
  --output data/processed/synthetic_sdn_ddos.npz \
  --steps 1200
```

A NumPy next-state prediction baseline was first used as an offline sanity
check:

```bash
python scripts/baseline_np_anomaly.py \
  --data data/processed/synthetic_sdn_ddos.npz \
  --output outputs/eval/synthetic_np_baseline.json \
  --window 8 \
  --ridge 10
```

## Result

```json
{
  "precision": 0.96,
  "recall": 1.0,
  "f1": 0.9795918367346939,
  "accuracy": 0.9916107382550335
}
```

This result only validates the offline experimental workflow. It should not be
reported as evidence for the final LeWM-SDN model.

After this initial check, PyTorch was installed in the local `.venv` and the
supervised MLP/CNN/LSTM baselines plus the actual Temporal GNN + SDN-JEPA
training/evaluation path were executed on the same synthetic dataset.

Supervised PyTorch baseline metrics:

```text
outputs/eval/torch_supervised_baselines.json
MLP:  precision=0.9875 recall=0.9875 f1=0.9875 accuracy=0.9950
CNN:  precision=0.9833 recall=0.9833 f1=0.9833 accuracy=0.9933
LSTM: precision=0.9833 recall=0.9833 f1=0.9833 accuracy=0.9933
```

The NumPy baseline comparison also includes a lightweight XGBoost-style
gradient-boosted decision-stump classifier:

```text
outputs/eval/baseline_comparison.json
Boosted stumps: precision=0.9796 recall=1.0000 f1=0.9897 accuracy=0.9958
```

The PyTorch Temporal GNN + SDN-JEPA smoke-run metrics are intentionally recorded
as weak rather than competitive:

```text
outputs/eval/torch_sdn_jepa_eval.json
precision=0.0753 recall=0.0032 f1=0.0062 accuracy=0.7911
```

Final scientific results still require a real dataset such as InSDN,
UNSW-NB15, CICIDS2017, or CICDDoS2019.
