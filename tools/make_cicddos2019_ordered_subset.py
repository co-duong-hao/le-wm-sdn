"""Create a CICDDoS2019 subset that preserves row order within each label.

This is intended for LeWM-SDN V1 experiments where consecutive rows are used as
next-state pairs. Unlike make_cicddos2019_subset.py, this script does not sample
rows randomly. It streams CSV files in sorted order and keeps the first N rows
seen for each label.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


LABEL_CANDIDATES = ("Label", "label", "Class", "class", "Attack", "attack")
DROP_COLUMNS = {
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Timestamp",
}


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(col).strip() for col in frame.columns]
    return frame


def detect_label_column(columns: list[str]) -> str | None:
    lower = {col.lower(): col for col in columns}
    for candidate in LABEL_CANDIDATES:
        found = lower.get(candidate.lower())
        if found:
            return found
    return None


def find_csv_files(root: Path, max_files: int | None) -> list[Path]:
    files = sorted(root.rglob("*.csv"))
    if max_files is not None:
        files = files[:max_files]
    return files


def clean_feature_frame(frame: pd.DataFrame, label_col: str) -> pd.DataFrame:
    labels = frame[label_col].astype(str).str.strip()
    candidates = frame.drop(columns=[label_col], errors="ignore")
    candidates = candidates.drop(columns=list(DROP_COLUMNS), errors="ignore")

    numeric = candidates.select_dtypes(include=["number"]).copy()
    numeric.replace([np.inf, -np.inf], np.nan, inplace=True)
    numeric.fillna(0.0, inplace=True)
    numeric["label"] = labels
    numeric["is_attack"] = (labels.str.upper() != "BENIGN").astype(np.int8)
    return numeric


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Folder containing CICDDoS2019 CSV files.")
    parser.add_argument("--out", required=True, help="Output subset CSV path.")
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-per-label", type=int, default=10_000)
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    out = Path(args.out).expanduser()
    files = find_csv_files(root, args.max_files)

    if not files:
        raise SystemExit(f"No CSV files found under: {root}")

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise SystemExit(f"Output already exists, refusing to overwrite: {out}")

    seen = Counter()
    written = Counter()
    header_written = False
    feature_columns: list[str] | None = None
    label_col: str | None = None

    print(f"Root: {root}")
    print(f"CSV files: {len(files)}")
    print(f"Output: {out}")
    print(f"Max per label: {args.max_per_label:,}")
    print("Mode: ordered first-N per label")
    print()

    for csv_path in files:
        print(f"[read] {csv_path.name}")
        for chunk in pd.read_csv(csv_path, chunksize=args.chunksize, low_memory=False):
            chunk = normalize_columns(chunk)

            if label_col is None:
                label_col = detect_label_column(list(chunk.columns))
                if label_col is None:
                    raise SystemExit("Could not find a label column.")

            cleaned = clean_feature_frame(chunk, label_col)

            if feature_columns is None:
                feature_columns = list(cleaned.columns)
            else:
                for col in feature_columns:
                    if col not in cleaned.columns:
                        cleaned[col] = 0.0
                cleaned = cleaned[feature_columns]

            for label, group in cleaned.groupby("label", sort=False):
                seen[label] += len(group)
                remaining = args.max_per_label - written[label]
                if remaining <= 0:
                    continue
                picked = group.iloc[:remaining]
                if picked.empty:
                    continue
                picked.to_csv(out, mode="a", header=not header_written, index=False)
                header_written = True
                written[label] += len(picked)

    print()
    print("[seen labels]")
    for label, count in seen.most_common():
        print(f"  {label}: {count:,}")

    print()
    print("[written labels]")
    for label, count in written.most_common():
        print(f"  {label}: {count:,}")

    print()
    print(f"Done: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

