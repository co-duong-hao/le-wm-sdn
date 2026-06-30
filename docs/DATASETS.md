# Dataset Guide for LeWM-SDN

This project needs dynamic SDN graph sequences rather than image frames. A useful
dataset should provide flow-level or packet-level network records that can be
aggregated into graph snapshots over time.

## Recommended Order

### 1. InSDN

Use InSDN as the main dataset.

- Best match for the project because it was built for SDN intrusion detection.
- Public CSV and PCAP data are available from UCD/ASEADOS.
- The CSV archive is small enough for quick experiments, while PCAP groups can
  be used later for richer flow extraction.
- Good fit for the first feasible stage:

```text
Temporal GNN + JEPA prediction + anomaly score
```

Source: <https://aseados.ucd.ie/datasets/SDN/>

### 2. CICDDoS2019

Use CICDDoS2019 as the main DDoS benchmark after InSDN.

- Contains many DDoS types, including DNS, LDAP, MSSQL, NetBIOS, NTP, SNMP,
  SSDP, UDP, UDP-Lag, SYN, TFTP, and WebDDoS.
- Has PCAP and CSV flow features.
- Not SDN-native, but strong for testing prediction-surprise and unknown attack
  detection.

Source: <https://www.unb.ca/cic/datasets/ddos-2019.html>

Offline conversion when CSV files are copied manually:

```bash
python scripts/prepare_flow_csv_graph.py \
  --input data/raw/cicddos2019 \
  --output data/processed/cicddos2019_graph.npz \
  --top-nodes 64 \
  --bin-seconds 1
```

### 2b. UNSW-NB15

Use UNSW-NB15 when SDN-native InSDN is unavailable but CSV files can be copied
offline. The repository includes `scripts/prepare_unsw_nb15.py`, which converts
the common public training/testing CSV files into the same `.npz` graph format.
When raw IP fields are unavailable, the converter builds a fixed heterogeneous
graph over protocol, service, state, and global traffic-summary nodes.

Expected placement:

```text
data/raw/unsw_nb15/UNSW_NB15_training-set.csv
data/raw/unsw_nb15/UNSW_NB15_testing-set.csv
```

Conversion:

```bash
python scripts/prepare_unsw_nb15.py \
  --input data/raw/unsw_nb15/UNSW_NB15_training-set.csv data/raw/unsw_nb15/UNSW_NB15_testing-set.csv \
  --output data/processed/unsw_nb15_graph.npz \
  --rows-per-step 500
```

### 2c. NF-UNSW-NB15

Use NF-UNSW-NB15 when a public UNSW-derived NetFlow benchmark is acceptable.
This project has already downloaded the public RDM archive and converted it to
the LeWM-SDN graph format.

Current local files:

```text
data/raw/nf_unsw_nb15/nf_unsw_nb15_rdm.zip
data/processed/nf_unsw_nb15_graph.npz
outputs/eval/nf_unsw_nb15_baseline_comparison.json
```

Conversion:

```bash
python scripts/prepare_flow_csv_graph.py \
  --input data/raw/nf_unsw_nb15/nf_unsw_nb15_rdm.zip \
  --output data/processed/nf_unsw_nb15_graph.npz \
  --extract-dir data/raw/nf_unsw_nb15/extracted \
  --top-nodes 64 \
  --rows-per-bin 1000
```

Evaluation:

```bash
python scripts/run_baseline_comparison.py \
  --data data/processed/nf_unsw_nb15_graph.npz \
  --output outputs/eval/nf_unsw_nb15_baseline_comparison.json \
  --history 8 \
  --latent-dim 16 \
  --split stratified \
  --eval-split test \
  --test-fraction 0.5
```

### 3. TON_IoT

Use TON_IoT to test generalization.

- Includes network traffic, IoT/IIoT telemetry, Windows traces, and Linux traces.
- Useful when extending the claim beyond pure DDoS to broader network anomalies.

Source: <https://research.unsw.edu.au/projects/toniot-datasets>

Offline conversion for network-flow CSV files:

```bash
python scripts/prepare_flow_csv_graph.py \
  --input data/raw/ton_iot \
  --output data/processed/ton_iot_graph.npz \
  --top-nodes 64 \
  --rows-per-bin 500
```

### 4. Bot-IoT

Use Bot-IoT if the project needs an IoT botnet/DDoS setting.

- Includes normal and botnet traffic with DoS, DDoS, scan, keylogging, and
  exfiltration.
- The full dataset is large, but the 5 percent subset is practical for early
  experiments.

Source: <https://research.unsw.edu.au/projects/bot-iot-dataset>

Offline conversion for CSV files with source/destination IP columns:

```bash
python scripts/prepare_flow_csv_graph.py \
  --input data/raw/bot_iot \
  --output data/processed/bot_iot_graph.npz \
  --top-nodes 64 \
  --rows-per-bin 500
```

## Target Format

The current code expects `.npz`, `.pt`, or `.pth` files with:

```text
node_features: (T, N, F_node) or (Episodes, T, N, F_node)
edge_index:    (2, E) or (E, 2)
edge_features: optional (T, E, F_edge), (Episodes, T, E, F_edge), or (E, F_edge)
label:         optional, 0 for normal and non-zero for attack
action:        optional mitigation action id/vector
```

## Graph Construction Policy

For flow CSV datasets such as InSDN:

- Node: IP address or endpoint.
- Edge: directed flow from source IP to destination IP.
- Time step: fixed wall-clock bin, for example 1 second.
- Node features:
  - incoming/outgoing flow count
  - incoming/outgoing packet count
  - incoming/outgoing byte count
  - protocol entropy or protocol count
  - port entropy or destination-port diversity
- Edge features:
  - flow count
  - packet count
  - byte count
  - mean duration
  - mean packet rate
  - attack ratio or binary attack flag for analysis only
- Label:
  - `0` when all flows in the time bin are normal
  - `1` when any attack flow appears in the time bin

The generic converter `scripts/prepare_flow_csv_graph.py` accepts a CSV file,
directory, or ZIP archive. It expects source/destination IP fields and supports
common timestamp, packet, byte, duration, protocol, port, and label column
names used by CICIDS/CICDDoS-style flow exports.

## Offline Alternative

If the InSDN website is unreachable, use the synthetic SDN/DDoS generator to
validate the full data path without internet access:

```bash
python scripts/generate_synthetic_sdn.py \
  --output data/processed/synthetic_sdn_ddos.npz \
  --steps 1200
```

This synthetic dataset is not a replacement for InSDN in final reporting. It is
useful for checking whether graph preparation, training, anomaly scoring, and
metrics run end to end.

Run the offline predictive-latent reference experiment:

```bash
python scripts/run_numpy_sdn_jepa_experiment.py \
  --data data/processed/synthetic_sdn_ddos.npz \
  --output outputs/eval/numpy_sdn_jepa_experiment.json \
  --history 8 \
  --latent-dim 16 \
  --ridge 1 \
  --alpha 1 \
  --beta 0.2
```

Run baseline comparison:

```bash
python scripts/run_baseline_comparison.py \
  --data data/processed/synthetic_sdn_ddos.npz \
  --output outputs/eval/baseline_comparison.json \
  --history 8 \
  --latent-dim 16 \
  --warmup 600
```

## InSDN Experiment Plan

1. Convert InSDN CSV to `data/processed/insdn_graph.npz`.

```bash
python scripts/prepare_insdn.py \
  --input data/raw/insdn/InSDN_DatasetCSV.zip \
  --output data/processed/insdn_graph.npz \
  --top-nodes 64 \
  --bin-seconds 1
```

2. Train only on normal windows:

```bash
python train.py data.path=data/processed/insdn_graph.npz data.num_steps=16
```

3. Evaluate anomaly score on all windows:

```bash
python eval.py checkpoint=outputs/checkpoints/lewm_sdn/lewm_sdn_best.pt data.path=data/processed/insdn_graph.npz data.num_steps=16
```

4. Report:
   - score mean/std
   - threshold
   - precision, recall, F1
   - surprise mean
   - phase drift mean

## Note on Mitigation Actions

Public datasets such as InSDN, CICDDoS2019, TON_IoT, and Bot-IoT usually do not
include true SDN mitigation actions like drop, reroute, or rate-limit. The MPC
part should therefore be validated later with a controlled Mininet + Ryu/ONOS
setup where actions and their consequences can be logged.
