import argparse, json, sys
from pathlib import Path
import pandas as pd

try:
    from great_expectations.dataset import PandasDataset
except Exception:
    print("[ERROR] pip install great-expectations>=0.18")
    sys.exit(1)

def jdump(o, p):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(o, f, indent=2, ensure_ascii=False)

def die(msg):
    print(f"[GE] {msg}")
    sys.exit(1)

def df_raw(root):
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

def df_meta(root):
    csvs = sorted(root.glob("*.csv"))
    if not csvs:
        return pd.DataFrame(columns=["filename","split","label"])
    out = []
    for c in csvs:
        d = pd.read_csv(c)
        d.columns = [str(x).lower() for x in d.columns]
        for k in ("filename","split","label"):
            if k not in d.columns: d[k] = pd.Series(dtype="object")
        out.append(d[["filename","split","label"]])
    return pd.concat(out, ignore_index=True)

def run_ge(df, exps, name, outdir):
    ge = PandasDataset(df)
    for e in exps:
        fn = getattr(ge, e["t"], None)
        if fn is None: die(f"Expectación desconocida: {e['t']}")
        fn(**e.get("k", {}))
    res = ge.validate()
    jdump(res, outdir / f"{name}.json")
    return res

def exps_raw():
    T = "expect_column_to_exist"; N = "expect_column_values_to_not_be_null"
    return [
        {"t": T, "k": {"column":"filepath"}}, {"t": T, "k": {"column":"filename"}},
        {"t": T, "k": {"column":"label"}}, {"t": T, "k": {"column":"ext"}},
        {"t": T, "k": {"column":"size_bytes"}},
        {"t": N, "k": {"column":"filepath"}}, {"t": N, "k": {"column":"filename"}},
        {"t": N, "k": {"column":"ext"}}, {"t": N, "k": {"column":"size_bytes"}},
        {"t":"expect_column_values_to_be_in_set","k":{"column":"ext","value_set":["mp4","avi","mov"],"mostly":0.99}},
        {"t":"expect_column_values_to_be_between","k":{"column":"size_bytes","min_value":1}},
        {"t":"expect_column_values_to_be_in_set","k":{"column":"label","value_set":["real","fake"]}},
        {"t":"expect_column_values_to_be_unique","k":{"column":"filepath"}},
    ]

def exps_meta():
    T = "expect_column_to_exist"; N = "expect_column_values_to_not_be_null"; O = "expect_column_values_to_be_of_type"
    return [
        {"t": T, "k": {"column":"filename"}}, {"t": T, "k": {"column":"split"}}, {"t": T, "k": {"column":"label"}},
        {"t": N, "k": {"column":"filename"}}, {"t": N, "k": {"column":"split"}}, {"t": N, "k": {"column":"label"}},
        {"t": O, "k": {"column":"filename","type_":"str"}},
        {"t": O, "k": {"column":"split","type_":"str"}},
        {"t": O, "k": {"column":"label","type_":"str"}},
        {"t":"expect_column_values_to_be_in_set","k":{"column":"split","value_set":["train","val"]}},
        {"t":"expect_column_values_to_be_in_set","k":{"column":"label","value_set":["real","fake"]}},
        {"t":"expect_column_values_to_be_unique","k":{"column":"filename"}},
        {"t":"expect_column_values_to_match_regex","k":{"column":"filename","regex":r".*\.(mp4|avi|mov)$","mostly":0.99}},
    ]

def validate_raw(data_dir, min_per_class):
    d = Path(data_dir) / "raw"
    if not d.exists(): die(f"No existe {d}.")
    df = df_raw(d)
    if df.empty: die("Sin ficheros en data/raw/{real,fake}.")
    cnt = df["label"].value_counts().to_dict()
    for lbl in ("real","fake"):
        if cnt.get(lbl,0) < min_per_class: die(f"Conteo insuficiente '{lbl}': {cnt.get(lbl,0)} < {min_per_class}")
    res = run_ge(df, exps_raw(), "raw_result", Path(data_dir)/"validation")
    if not res.get("success", False): die("RAW validation FAILED (data/validation/raw_result.json)")
    print("[GE] RAW validation PASSED")

def validate_metadata(data_dir):
    d = Path(data_dir) / "metadata"
    if not d.exists(): die(f"No existe {d}.")
    df = df_meta(d)
    if df.empty: die("Sin CSVs en data/metadata/*.csv.")
    res = run_ge(df, exps_meta(), "metadata_result", Path(data_dir)/"validation")
    if not res.get("success", False): die("Metadata validation FAILED (data/validation/metadata_result.json)")
    if (df.groupby("filename")["split"].nunique() > 1).any(): die("Leakage: un filename aparece en >1 split.")
    print("[GE] Metadata validation PASSED")

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("validate-raw")
    r.add_argument("--data-dir", type=Path, default=Path("data"))
    r.add_argument("--min-per-class", type=int, default=10)
    m = sub.add_parser("validate-metadata")
    m.add_argument("--data-dir", type=Path, default=Path("data"))
    a = p.parse_args()
    if a.cmd == "validate-raw": validate_raw(a.data_dir, a.min_per_class)
    else: validate_metadata(a.data_dir)

if __name__ == "__main__":
    main()

