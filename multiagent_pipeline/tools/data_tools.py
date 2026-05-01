"""Pure tools for the DataAgent.

All functions are deterministic pandas operations, with no LLM or global state.
Testable in isolation with pytest.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DATA_AGENT_MANIFEST = _PROJECT_ROOT / "data" / "processed" / "data_agent_output.json"


def load_last_perimeter() -> dict:
    """Reads the perimeter from the last DataAgent run from the manifest on disk.

    Useful in __main__ blocks of standalone agents: instead of hardcoding
    {"anno": 2024}, it always uses the last interactively selected perimeter.
    Returns {} if the manifest does not exist or is unreadable.
    """
    if not _DATA_AGENT_MANIFEST.exists():
        return {}
    try:
        manifest = json.loads(_DATA_AGENT_MANIFEST.read_text())
        raw = manifest.get("perimeter", {})
        # Filter out None values (fields not set by the Perimeter Pydantic model)
        return {k: v for k, v in raw.items() if v is not None}
    except Exception:
        return {}


def load_dataset(path: Path | str) -> pd.DataFrame:
    """Loads the merged dataset produced by the classical preprocessing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


# Mapping: perimeter key -> actual column name in the CSV.
# Keys MUST match the field names of state.Perimeter (Pydantic),
# which is the DataAgent's contract.
_PERIMETER_COLS = {
    "anno":               "ANNO_PARTENZA",
    "mese":               "MESE_PARTENZA",      # extra (not in Perimeter, but useful)
    "aeroporto_partenza": "AREOPORTO_PARTENZA",
    "aeroporto_arrivo":   "AREOPORTO_ARRIVO",
    "paese_partenza":     "PAESE_PART",
    "zona":               "ZONA",
}


def filter_by_perimeter(df: pd.DataFrame, perimeter: Optional[dict]) -> pd.DataFrame:
    """Filters the DataFrame by applying the perimeter constraints.

    Unspecified keys -> no filter on that dimension.
    Returns a copy (does not modify the input).
    """
    if not perimeter:
        return df.copy()

    out = df
    for key, value in perimeter.items():
        if value is None:
            continue
        col = _PERIMETER_COLS.get(key)
        if col is None:
            raise KeyError(f"Unknown perimeter key: {key}")
        if col not in out.columns:
            raise KeyError(f"Column {col} not present in the dataset")
        # Case-insensitive comparison on string columns for consistency with DataAgent.
        if pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col]):
            out = out[out[col].astype(str).str.upper() == str(value).upper()]
        else:
            out = out[out[col] == value]
    return out.copy()


def get_dataset_stats(df: pd.DataFrame) -> dict:
    """Descriptive statistics of the filtered DataFrame (for the report and logs)."""
    stats = {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
    }
    if "ANNO_PARTENZA" in df.columns and len(df):
        stats["anni"] = sorted(df["ANNO_PARTENZA"].dropna().unique().tolist())
    if "AREOPORTO_PARTENZA" in df.columns:
        stats["n_aeroporti_part"] = int(df["AREOPORTO_PARTENZA"].nunique())
    if "AREOPORTO_ARRIVO" in df.columns:
        stats["n_aeroporti_arr"] = int(df["AREOPORTO_ARRIVO"].nunique())
    if "PAESE_ARR" in df.columns:
        stats["n_paesi_arr"] = int(df["PAESE_ARR"].nunique())
    return stats
