# LeWM-SDN

LeWM-SDN adapts the core idea of LeWorldModel to DDoS detection and mitigation in
Software-Defined Networking. It does not reuse the original pixel/robot model
directly. Instead, it learns a latent world model of SDN network dynamics from
dynamic graph snapshots.

The central question changes from:

```text
Is this packet/flow attack or normal?
```

to:

```text
Is the next network state still plausible under the normal latent dynamics?
```

## Architecture

The original LeWM mapping

```text
z_t = enc(o_t)
z_hat_{t+1} = pred(z_t, a_t)
```

is translated to SDN as:

```text
G_t = (V_t, E_t, X_t, A_t)
z_t = G-enc(G_t)
z_hat_{t+1} = pred(z_t, u_t)
```

where `G_t` is the SDN graph snapshot and `u_t` is a mitigation action such as
no-op, drop-flow, rate-limit, reroute, or install-filter.

Implemented components:

- `TemporalGraphEncoder` replaces the ViT image encoder with graph message
  passing plus temporal recurrence.
- `SDNJEPA` predicts future SDN latent states.
- The latent vector is split into `z_prog` and `z_cont`.
- `SIGReg` is applied only to `z_cont`, not the full latent vector.
- Anomaly scoring combines prediction surprise and latent phase drift:

```text
A_t = alpha * ||z_hat_{t+1} - z_{t+1}||^2 + beta * |wrap(theta_t - theta_{t-1})|
theta_t = atan2(z_prog[1], z_prog[0])
```

The code also includes a latent defense-cost hook for future MPC/CEM mitigation
planning. This is not part of the evaluated paper results because the public
datasets used here do not contain logged mitigation actions:

```text
C = congestion_weight * Congestion(z)
  + packet_loss_weight * PacketLoss(z)
  + mitigation_weight * Cost(u)
```

## Installation

The SDN pipeline only needs the general PyTorch stack:

```bash
pip install torch hydra-core omegaconf numpy einops
```

## Data Format

Training and evaluation use `.npz`, `.pt`, or `.pth` files with tensor-like
arrays.

Required:

```text
node_features: (T, N, F_node) or (Episodes, T, N, F_node)
edge_index:    (2, E) or (E, 2)
```

Optional:

```text
edge_features:      (T, E, F_edge), (Episodes, T, E, F_edge), or (E, F_edge)
action:             discrete ids (T) / (Episodes, T), or vectors (..., A)
mitigation_action:  same as action
label:              0 for normal, non-zero for attack
node_mask:          (T, N) or (Episodes, T, N)
```

Typical node features include CPU, memory, flow count, queue length, packet
rate, and controller load. Typical edge features include bandwidth, packet
rate, packet loss, and link utilization.

## Training

By default, training filters to normal windows when `label` is available:

```bash
python train.py data.path=/path/to/sdn_train.npz
```

Useful overrides:

```bash
python train.py \
  data.path=/path/to/sdn_train.npz \
  history_size=8 \
  data.num_steps=16 \
  embed_dim=192 \
  prog_dim=32
```

Checkpoints are written to:

```text
outputs/checkpoints/lewm_sdn/
```

The training objective is:

```text
L = L_pred(z)
  + lambda_s * SIGReg(z_cont)
  + lambda_r * L_triplet(z_prog)
  + lambda_st * L_straight(z)
```

## Evaluation

Run anomaly scoring on a labeled or unlabeled SDN sequence:

```bash
python eval.py \
  checkpoint=outputs/checkpoints/lewm_sdn/lewm_sdn_best.pt \
  data.path=/path/to/sdn_eval.npz \
  data.num_steps=16
```

The output JSON contains the threshold, aggregate score statistics, number of
flagged transitions, and precision/recall/F1 when labels are present.

Default output:

```text
outputs/eval/sdn_anomaly.json
```

## Offline Experiment

When public datasets cannot be downloaded, the repository includes a runnable
offline SDN/DDoS experiment:

```bash
python scripts/generate_synthetic_sdn.py --output data/processed/synthetic_sdn_ddos.npz --steps 1200 --seed 3072
python scripts/run_numpy_sdn_jepa_experiment.py --data data/processed/synthetic_sdn_ddos.npz --output outputs/eval/numpy_sdn_jepa_experiment.json --history 8 --latent-dim 16 --ridge 1 --alpha 1 --beta 0.2
```

The repository tracks the small recorded metric artifacts:

```text
outputs/eval/baseline_comparison.json
outputs/eval/nf_unsw_nb15_baseline_comparison.json
outputs/eval/torch_supervised_baselines.json
outputs/eval/torch_sdn_jepa_eval.json
```

The main positive result is in the anomaly-only setting, where attack labels
are not used during training. The standalone NumPy predictive-latent reference
reaches F1 = 0.9796 on the generated dataset, slightly above one-class feature
distance at F1 = 0.9717 and far above raw-feature ridge surprise at F1 = 0.2644.
The stricter aligned baseline-comparison protocol reports F1 = 0.9505.
A lightweight supervised NumPy gradient-boosting baseline reaches F1 = 0.9897.
Supervised PyTorch baselines reach F1 = 0.9875 for MLP and F1 = 0.9833 for
CNN/LSTM, while the minimal PyTorch Temporal GNN + SDN-JEPA sanity check reaches
only F1 = 0.0062. This is recorded deliberately to show that the neural code
path executes, not to claim a tuned neural benchmark.

The repository also includes the recorded public NF-UNSW-NB15 results:

```text
outputs/eval/nf_unsw_nb15_public_experiment.md
outputs/eval/nf_unsw_torch_sdn_jepa_eval.json
main.pdf
outputs/eval/paper_build_report.md
```

Raw public archives, generated `.npz` tensors, and checkpoint weights are kept
out of Git. Recreate them with the scripts in `scripts/` and the dataset
instructions in `docs/DATASETS.md`.

On this public split, supervised logistic regression and NumPy MLP reach
F1 = 0.9908, while the current LeWM-SDN latent surprise + phase reference only
reaches F1 = 0.0841. The public PyTorch Temporal GNN + SDN-JEPA sanity check
trains successfully but reaches F1 = 0.0000 with the default normal-score
threshold. This is a negative but important public benchmark result and should
be read as a calibration/tuning limitation, not as evidence that the controlled
anomaly-only result transfers yet.

The current recorded report is:

```text
outputs/eval/project_success_report.md
docs/EXPERT_FEEDBACK_AUDIT.md
outputs/eval/paper_build_report.md
```

## Source Map

- `module.py`: Temporal GNN, mitigation action encoder, transformer predictor,
  and SIGReg.
- `jepa.py`: `SDNJEPA`, phase drift, anomaly score, straightening loss, and
  defense-cost hook.
- `sdn_data.py`: dynamic SDN graph window dataset.
- `train.py`: SDN-JEPA training loop.
- `eval.py`: DDoS anomaly scoring and metrics.
- `scripts/run_numpy_sdn_jepa_experiment.py`: offline NumPy reference experiment
  for the predictive latent anomaly score.
- `scripts/run_torch_supervised_baselines.py`: supervised MLP/CNN/LSTM baseline
  comparison on graph-feature windows.
- `scripts/prepare_flow_csv_graph.py`: generic public flow-CSV converter for
  CICIDS/CICDDoS/TON/Bot-IoT-style files with endpoint IP columns.
- `config/train/model/sdn_jepa.yaml`: SDN model config.
- `config/train/data/sdn.yaml`: SDN dataset config.
- `config/eval/sdn.yaml`: anomaly evaluation config.

## Research Status

This repository now implements the feasible first stage of the document:

```text
Temporal GNN + JEPA prediction + anomaly score
```

The SD-JEPA split and phase anomaly are present. MPC/CEM mitigation planning is
represented only as a future-work latent rollout and defense-cost hook; it
should be validated later in a controlled SDN testbed with logged actions and
post-action states.
