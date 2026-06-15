"""Inspect CICDDoS2019 CSV files before building the training pipeline.

This script is intentionally read-only. It scans CSV files, detects the label
column, estimates label distributions, and prints a compact feature summary.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd


LABEL_CANDIDATES = ("Label", "label", "Class", "class", "Attack", "attack")


def find_csv_files(root: Path, max_files: int | None) -> list[Path]:
    files = sorted(root.rglob("*.csv"))
    if max_files is not None:
        files = files[:max_files]
    return files


def detect_label_column(columns: list[str]) -> str | None:
    normalized = {col.strip(): col for col in columns}
    for candidate in LABEL_CANDIDATES:
        if candidate in normalized:
            return normalized[candidate]
    lower = {col.strip().lower(): col for col in columns}
    for candidate in LABEL_CANDIDATES:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def inspect_file(path: Path, chunksize: int) -> dict:
    row_count = 0
    label_counts: Counter[str] = Counter()
    columns: list[str] | None = None
    label_col: str | None = None
    numeric_columns: set[str] = set()

    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        if columns is None:
            columns = list(chunk.columns)
            label_col = detect_label_column(columns)

        row_count += len(chunk)

        if label_col and label_col in chunk:
            values = chunk[label_col].astype(str).str.strip()
            label_counts.update(values.value_counts(dropna=False).to_dict())

        for col in chunk.columns:
            if col == label_col:
                continue
            if pd.api.types.is_numeric_dtype(chunk[col]):
                numeric_columns.add(col)

    return {
        "path": path,
        "rows": row_count,
        "columns": columns or [],
        "label_col": label_col,
        "label_counts": label_counts,
        "numeric_columns": numeric_columns,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Folder containing CSV files.")
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    if not root.exists():
        raise SystemExit(f"Dataset folder does not exist: {root}")

    files = find_csv_files(root, args.max_files)
    if not files:
        raise SystemExit(f"No CSV files found under: {root}")

    print(f"Root: {root}")
    print(f"CSV files scanned: {len(files)}")
    print(f"Chunksize: {args.chunksize}")
    print()

    total_rows = 0
    total_labels: Counter[str] = Counter()
    all_numeric: set[str] = set()
    first_columns: list[str] | None = None
    detected_label: str | None = None

    for file_path in files:
        result = inspect_file(file_path, args.chunksize)
        total_rows += result["rows"]
        total_labels.update(result["label_counts"])
        all_numeric.update(result["numeric_columns"])
        first_columns = first_columns or result["columns"]
        detected_label = detected_label or result["label_col"]

        print(f"[file] {file_path.name}")
        print(f"  rows: {result['rows']:,}")
        print(f"  columns: {len(result['columns'])}")
        print(f"  label column: {result['label_col'] or 'not found'}")
        if result["label_counts"]:
            top_labels = result["label_counts"].most_common(8)
            print("  top labels:")
            for label, count in top_labels:
                print(f"    {label}: {count:,}")
        print()

    print("[summary]")
    print(f"total rows: {total_rows:,}")
    print(f"detected label column: {detected_label or 'not found'}")
    print(f"numeric feature columns: {len(all_numeric)}")

    if first_columns:
        preview = ", ".join(first_columns[:12])
        suffix = " ..." if len(first_columns) > 12 else ""
        print(f"first columns: {preview}{suffix}")

    if total_labels:
        print("label distribution:")
        for label, count in total_labels.most_common():
            print(f"  {label}: {count:,}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

