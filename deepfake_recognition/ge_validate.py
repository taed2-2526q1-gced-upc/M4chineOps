
#   python -m deepfake_recognition.validation.ge_validate validate-raw
#   python -m deepfake_recognition.validation.ge_validate validate-metadata
# Opcionales:
#   --data-dir data               (raíz de datos)
#   --min-per-class 10            (mínimo de vídeos por clase en RAW)

from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import pandas as pd

try:
    from great_expectations.dataset import PandasDataset
except Exception as e:
    print("[ERROR] Necesitas instalar great-expectations (pip install great-expectations>=0.18).")
    raise


# Helpers


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def save_json(obj, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def fail(msg: str) -> None:
    print(f"[GE]  {msg}")
    sys.exit(1)

def ok(msg: str) -> None:
    print(f"[GE]  {msg}")

# RAW: construir DataFrame a partir de ficheros


def build_raw_df(raw_dir: Path) -> pd.DataFrame:
    rows = []
    for label in ["real", "fake"]:
        label_dir = raw_dir / label
        if not label_dir.exists():
            continue
        for p in label_dir.rglob("*"):
            if p.is_file():
                rows.append({
                    "filepath": str(p),
                    "filename": p.name,
                    "label": label,
                    "ext": p.suffix.lower().lstrip("."),
                    "size_bytes": p.stat().st_size
                })
    if not rows:
        return pd.DataFrame(columns=["filepath","filename","label","ext","size_bytes"])
    return pd.DataFrame(rows)


# METADATA: leer CSV(s) a DataFrame


def build_metadata_df(metadata_dir: Path) -> pd.DataFrame:
    csvs = sorted(metadata_dir.glob("*.csv"))
    if not csvs:
        return pd.DataFrame(columns=["filename","split","label"])
    dfs = []
    for c in csvs:
        df = pd.read_csv(c)
        # normaliza nombres de columnas si llegan con mayúsculas/variantes
        colmap = {k: k.lower() for k in df.columns}
        df.columns = [colmap.get(c, c).lower() for c in df.columns]
        # Asegura columnas requeridas
        for needed in ["filename", "split", "label"]:
            if needed not in df.columns:
                df[needed] = pd.Series(dtype="object")
        df = df[["filename","split","label"]]
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


# Ejecutar Expectativas GE


def run_expectations(df: pd.DataFrame, expectations: list[dict], result_name: str, out_dir: Path) -> dict:
    ge_df = PandasDataset(df.copy())
    for exp in expectations:
        fn_name = exp["type"]
        kwargs = exp.get("kwargs", {})
        fn = getattr(ge_df, fn_name, None)
        if fn is None:
            fail(f"Expectación desconocida: {fn_name}")
        fn(**kwargs)
    results = ge_df.validate()
    save_json(results, out_dir / f"{result_name}.json")
    return results


# Suites (labels = real/fake)


def expectations_raw() -> list[dict]:
    return [
        {"type": "expect_column_to_exist", "kwargs": {"column": "filepath"}},
        {"type": "expect_column_to_exist", "kwargs": {"column": "filename"}},
        {"type": "expect_column_to_exist", "kwargs": {"column": "label"}},
        {"type": "expect_column_to_exist", "kwargs": {"column": "ext"}},
        {"type": "expect_column_to_exist", "kwargs": {"column": "size_bytes"}},

        {"type": "expect_column_values_to_not_be_null", "kwargs": {"column": "filepath"}},
        {"type": "expect_column_values_to_not_be_null", "kwargs": {"column": "filename"}},
        {"type": "expect_column_values_to_not_be_null", "kwargs": {"column": "ext"}},
        {"type": "expect_column_values_to_not_be_null", "kwargs": {"column": "size_bytes"}},

        {"type": "expect_column_values_to_be_in_set",
         "kwargs": {"column": "ext", "value_set": ["mp4","avi","mov"], "mostly": 0.99}},
        {"type": "expect_column_values_to_be_between",
         "kwargs": {"column": "size_bytes", "min_value": 1}},
        {"type": "expect_column_values_to_be_in_set",
         "kwargs": {"column": "label", "value_set": ["real","fake"], "mostly": 1.0}},

        {"type": "expect_column_values_to_be_unique", "kwargs": {"column": "filepath"}},
    ]

def expectations_metadata() -> list[dict]:
    return [
        {"type": "expect_column_to_exist", "kwargs": {"column": "filename"}},
        {"type": "expect_column_to_exist", "kwargs": {"column": "split"}},
        {"type": "expect_column_to_exist", "kwargs": {"column": "label"}},

        {"type": "expect_column_values_to_not_be_null", "kwargs": {"column": "filename"}},
        {"type": "expect_column_values_to_not_be_null", "kwargs": {"column": "split"}},
        {"type": "expect_column_values_to_not_be_null", "kwargs": {"column": "label"}},

        {"type": "expect_column_values_to_be_of_type", "kwargs": {"column": "filename", "type_": "str"}},
        {"type": "expect_column_values_to_be_of_type", "kwargs": {"column": "split", "type_": "str"}},
        {"type": "expect_column_values_to_be_of_type", "kwargs": {"column": "label", "type_": "str"}},

        {"type": "expect_column_values_to_be_in_set", "kwargs": {"column": "split", "value_set": ["train", "val"]}},
        {"type": "expect_column_values_to_be_in_set", "kwargs": {"column": "label", "value_set": ["real", "fake"]}},

        {"type": "expect_column_values_to_be_unique", "kwargs": {"column": "filename"}},

        {"type": "expect_column_values_to_match_regex",
         "kwargs": {"column": "filename", "regex": r".*\.(mp4|avi|mov)$", "mostly": 0.99}},
    ]


# Validadores


def validate_raw(data_dir: Path, min_per_class: int) -> None:
    raw_dir = data_dir / "raw"
    if not raw_dir.exists():
        fail(f"No existe {raw_dir}. ¿Has ejecutado la etapa 'download'?")

    df = build_raw_df(raw_dir)
    if df.empty:
        fail("No se han encontrado ficheros en data/raw/{real,fake}.")

    # Conteo mínimo por clase
    counts = df["label"].value_counts().to_dict()
    for label in ["real","fake"]:
        if counts.get(label, 0) < min_per_class:
            fail(f"Conteo insuficiente para clase '{label}': {counts.get(label,0)} < {min_per_class}")

    # Expectativas GE
    results = run_expectations(
        df=df,
        expectations=expectations_raw(),
        result_name="raw_result",
        out_dir=data_dir / "validation"
    )
    if not results.get("success", False):
        fail("RAW validation FAILED (ver data/validation/raw_result.json)")
    ok("RAW validation PASSED")

def validate_metadata(data_dir: Path) -> None:
    meta_dir = data_dir / "metadata"
    if not meta_dir.exists():
        fail(f"No existe {meta_dir}. ¿Has ejecutado la etapa 'sampling_and_metadata'?")

    df = build_metadata_df(meta_dir)
    if df.empty:
        fail("No se han encontrado CSVs de metadatos en data/metadata/*.csv.")

    # Expectativas GE
    results = run_expectations(
        df=df,
        expectations=expectations_metadata(),
        result_name="metadata_result",
        out_dir=data_dir / "validation"
    )
    if not results.get("success", False):
        fail("Metadata validation FAILED (ver data/validation/metadata_result.json)")

    # Anti-leakage: un filename no puede estar en >1 split
    leakage = (df.groupby("filename")["split"].nunique() > 1)
    n_bad = int(leakage.sum())
    if n_bad > 0:
        fail(f"Leakage detectado: {n_bad} filename(s) aparecen en más de un split.")

    ok("Metadata validation PASSED")


# CLI


def main():
    parser = argparse.ArgumentParser(description="Great Expectations validators (real/fake).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_raw = sub.add_parser("validate-raw", help="Valida data/raw (extensiones, tamaño, conteos).")
    p_raw.add_argument("--data-dir", type=Path, default=Path("data"), help="Raíz de datos (por defecto: data)")
    p_raw.add_argument("--min-per-class", type=int, default=10, help="Mínimo de vídeos por clase (por defecto: 10)")

    p_meta = sub.add_parser("validate-metadata", help="Valida data/metadata/*.csv (esquema y consistencia).")
    p_meta.add_argument("--data-dir", type=Path, default=Path("data"), help="Raíz de datos (por defecto: data)")

    args = parser.parse_args()

    if args.cmd == "validate-raw":
        validate_raw(args.data_dir, args.min_per_class)
    elif args.cmd == "validate-metadata":
        validate_metadata(args.data_dir)
    else:
        parser.print_help()
        sys.exit(2)

if __name__ == "__main__":
    main()
