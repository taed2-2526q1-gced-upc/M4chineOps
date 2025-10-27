#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
import pandas as pd

try:
    from great_expectations.dataset import PandasDataset
except Exception as e:
    print(f"[ERROR] pip install great-expectations>=0.18: {e}")
    sys.exit(1)

def jdump(obj, path):
    from pathlib import Path
    import json
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    try:
        obj = obj.to_json_dict()
    except Exception:
        pass
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)

def die(msg: str):
    print(f"[GE] {msg}")
    sys.exit(1)

def df_meta(root: Path) -> pd.DataFrame:
    csvs = sorted(root.glob("*.csv"))
    if not csvs:
        return pd.DataFrame(columns=["filename","split","label"])
    out = []
    for c in csvs:
        d = pd.read_csv(c)
        d.columns = [str(x).lower() for x in d.columns]
        # asegurar columnas mínimas
        for k in ("filename","split","label"):
            if k not in d.columns:
                d[k] = pd.Series(dtype="object")
        out.append(d[["filename","split","label"]])
    return pd.concat(out, ignore_index=True)

def exps_meta():
    T = "expect_column_to_exist"
    N = "expect_column_values_to_not_be_null"
    O = "expect_column_values_to_be_of_type"
    return [
        {"t": T, "k": {"column":"filename"}},
        {"t": T, "k": {"column":"split"}},
        {"t": T, "k": {"column":"label"}},
        {"t": N, "k": {"column":"filename"}},
        {"t": N, "k": {"column":"split"}},
        {"t": N, "k": {"column":"label"}},
        {"t": O, "k": {"column":"filename","type_":"str"}},
        {"t": O, "k": {"column":"split","type_":"str"}},
        {"t": O, "k": {"column":"label","type_":"str"}},
        {"t":"expect_column_values_to_be_in_set","k":{"column":"split","value_set":["train","val"]}},
        {"t":"expect_column_values_to_be_in_set","k":{"column":"label","value_set":["real","fake"]}},
        {"t":"expect_column_values_to_be_unique","k":{"column":"filename"}},
        {"t":"expect_column_values_to_match_regex","k":{"column":"filename","regex":r".*\.(mp4|avi|mov)$","mostly":0.99}},
    ]

def run_ge(df: pd.DataFrame, exps, out_json: Path):
    ge = PandasDataset(df)
    for e in exps:
        fn = getattr(ge, e["t"], None)
        if fn is None:
            die(f"Expectación desconocida: {e['t']}")
        fn(**e.get("k", {}))
    res = ge.validate()
    jdump(res.to_json_dict(), out_json)
    return res

def validate_metadata(data_dir: Path):
    d = data_dir / "metadata"
    if not d.exists():
        die(f"{d} does not exist.")
    df = df_meta(d)
    if df.empty:
        die("No files in data/metadata/*.csv.")
    out = data_dir / "validation" / "metadata_result.json"
    res = run_ge(df, exps_meta(), out)
    try:
        res = res.to_json_dict()
    except Exception:
        pass
    success = res.get('success', False) if isinstance(res, dict) else getattr(res, 'success', False)
    if not success:
        die("Metadata validation FAILED (data/validation/metadata_result.json)")
    # chequeo de leakage: mismo filename en >1 split
    if (df.groupby("filename")["split"].nunique() > 1).any():
        die("Leakage: un filename aparece en >1 split.")
    print("[GE] Metadata validation PASSED")

def main():
    p = argparse.ArgumentParser(description="Validate metadata CSVs with Great Expectations.")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    a = p.parse_args()
    validate_metadata(a.data_dir)

if __name__ == "__main__":
    main()
