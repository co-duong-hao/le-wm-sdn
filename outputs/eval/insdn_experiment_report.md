# InSDN Experiment Report

Date: 2026-06-29

## Status

The InSDN experiment was prepared but not completed because the InSDN dataset
file is not available in the current environment. PyTorch is now available in
the local `.venv`, and the Temporal GNN + SDN-JEPA training/evaluation path has
been smoke-tested on the synthetic SDN graph dataset.

## Completed

- Created dataset guide: `docs/DATASETS.md`
- Created InSDN converter: `scripts/prepare_insdn.py`
- Created data folders:
  - `data/raw/insdn`
  - `data/processed`
  - `outputs/eval`
- Verified converter syntax with `py_compile`

## Download Attempt

Target file:

```text
https://aseados.ucd.ie/datasets/SDN/InSDN_DatasetCSV.zip
```

Commands attempted:

```powershell
Invoke-WebRequest -Uri "https://aseados.ucd.ie/datasets/SDN/InSDN_DatasetCSV.zip" -OutFile "data\raw\insdn\InSDN_DatasetCSV.zip"
curl.exe -L -k "https://aseados.ucd.ie/datasets/SDN/InSDN_DatasetCSV.zip" -o "data\raw\insdn\InSDN_DatasetCSV.zip"
curl.exe -L "http://aseados.ucd.ie/datasets/SDN/InSDN_DatasetCSV.zip" -o "data\raw\insdn\InSDN_DatasetCSV.zip"
```

Observed error:

```text
Failed to connect to aseados.ucd.ie port 443/80
```

## Runtime Check

Initial Python check through `uv`:

```text
C:\Users\ADMIN\AppData\Roaming\uv\python\cpython-3.13-windows-x86_64-none\python.exe
```

Observed error:

```text
ModuleNotFoundError: No module named 'torch'
```

Resolution:

```text
.venv\Scripts\python.exe
```

The local virtual environment now has PyTorch, Hydra, OmegaConf, NumPy, and
einops. The converter can run once the InSDN CSV/ZIP file is available, and
training/evaluation can run through `.venv\Scripts\python.exe`.

## Resume Commands

After placing `InSDN_DatasetCSV.zip` at:

```text
data/raw/insdn/InSDN_DatasetCSV.zip
```

run:

```bash
python scripts/prepare_insdn.py \
  --input data/raw/insdn/InSDN_DatasetCSV.zip \
  --output data/processed/insdn_graph.npz \
  --top-nodes 64 \
  --bin-seconds 1
```

Then train:

```bash
.venv\Scripts\python.exe train.py data.path=data/processed/insdn_graph.npz data.num_steps=16
```

Then evaluate:

```bash
.venv\Scripts\python.exe eval.py \
  checkpoint=outputs/checkpoints/lewm_sdn/lewm_sdn_best.pt \
  data.path=data/processed/insdn_graph.npz \
  data.num_steps=16
```

## Offline Alternative Executed

Because web access to InSDN failed, an offline synthetic SDN/DDoS dataset was
generated instead:

```text
data/processed/synthetic_sdn_ddos.npz
```

The NumPy baseline result is stored at:

```text
outputs/eval/synthetic_np_baseline.json
```

The current consolidated report, including baseline comparison and the PyTorch
Temporal GNN + SDN-JEPA smoke run, is stored at:

```text
outputs/eval/project_success_report.md
```

Detailed note:

```text
outputs/eval/offline_synthetic_experiment.md
```
