#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys

import pandas as pd

try:
    from great_expectations.dataset import PandasDataset
except Exception as e:
    print(f"[ERROR] pip install great-expectations>=0.18: {e}")
    sys.exit(1)

def jdump(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def die(msg: str):
    print(f"[GE] {msg}")
    sys.exit(1)

def df_raw(root: Path) -> pd.DataFrame:
    rows = []
    for lbl in ("real", "fake"):
        d = root / lbl
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if p.is_file():
                rows.append({
                    "filepath": str(p),
                    "filename": p.name,
                    "label": lbl,
                    "ext": p.suffix.lower().lstrip("."),
                    "size_bytes": p.stat().st_size,
                })
    return pd.DataFrame(rows, columns=["filepath","filename","label","ext","size_bytes"])

def exps_raw():
    T = "expect_column_to_exist"
    N = "expect_column_values_to_not_be_null"
    return [
        {"t": T, "k": {"column":"filepath"}},
        {"t": T, "k": {"column":"filename"}},
        {"t": T, "k": {"column":"label"}},
        {"t": T, "k": {"column":"ext"}},
        {"t": T, "k": {"column":"size_bytes"}},
        {"t": N, "k": {"column":"filepath"}},
        {"t": N, "k": {"column":"filename"}},
        {"t": N, "k": {"column":"ext"}},
        {"t": N, "k": {"column":"size_bytes"}},
        {"t":"expect_column_values_to_be_in_set","k":{"column":"ext","value_set":["mp4","avi","mov"],"mostly":0.99}},
        {"t":"expect_column_values_to_be_between","k":{"column":"size_bytes","min_value":1}},
        {"t":"expect_column_values_to_be_in_set","k":{"column":"label","value_set":["real","fake"]}},
        {"t":"expect_column_values_to_be_unique","k":{"column":"filepath"}},
    ]

def run_ge(df: pd.DataFrame, exps, out_json: Path):
    ge = PandasDataset(df)
    for e in exps:
        fn = getattr(ge, e["t"], None)
        if fn is None:
            die(f"Expectación desconocida: {e['t']}")
        fn(**e.get("k", {}))
    res = ge.validate()
    jdump(res, out_json)
    return res

def validate_raw(data_dir: Path, min_per_class: int):
    d = data_dir / "raw"
    if not d.exists():
        die(f"{d} does not exist.")
    df = df_raw(d)
    if df.empty:
        die("No files in data/raw/{real,fake}.")
    cnt = df["label"].value_counts().to_dict()
    for lbl in ("real","fake"):
        if cnt.get(lbl, 0) < min_per_class:
            die(f"Not enough videos sampled per class '{lbl}': {cnt.get(lbl,0)} < {min_per_class}")
    out = data_dir / "validation" / "raw_result.json"
    res = run_ge(df, exps_raw(), out)
    if not res.get("success", False):
        die("RAW validation FAILED (data/validation/raw_result.json)")
    print("[GE] RAW validation PASSED")

def main():
    p = argparse.ArgumentParser(description="Validate raw video inventory with Great Expectations.")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--min-per-class", type=int, default=10)
    a = p.parse_args()
    validate_raw(a.data_dir, a.min_per_class)

if __name__ == "__main__":
    main()
