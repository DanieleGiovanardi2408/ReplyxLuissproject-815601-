"""
preprocessing.py
────────────────
Cleaning and normalization of the ALLARMI and TIPOLOGIA_VIAGGIATORE datasets.
This script is SHARED between both pipelines (classical and multi-agent).

Cleans and normalizes the two raw datasets, then merges them by route and date.
Handles missing values, encoding errors, mixed date formats, and domain constraints.

Input:
    data/raw/ALLARMI.csv
    data/raw/TIPOLOGIA_VIAGGIATORE.csv

Output:
    data/processed/allarmi_clean.csv
    data/processed/viaggiatori_clean.csv
    data/processed/dataset_merged.csv

Usage:
    python shared/preprocessing.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent
RAW_DIR       = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

ALLARMI_PATH     = RAW_DIR / "ALLARMI.csv"
VIAGGIATORI_PATH = RAW_DIR / "TIPOLOGIA_VIAGGIATORE.csv"

# ── Duplicate columns to drop ─────────────────────────────────────────────────
COLS_DROP_ALLARMI = [
    "Paese Partenza", "CODICE PAESE ARR", "3zona", "paese%arr", "tot voli",
]
COLS_DROP_VIAGGIATORI = [
    "Tipo Documento", "FASCIA ETA", "3nazionalita", "compagnia%aerea", "num volo",
]

# ── NULL: all variants found in the raw dataset ───────────────────────────────
NULL_VALUES = [
    "N.D.", "n.d.", "ND", "N/D", "N/A", "n/a", "?", "??", "???",
    "//", "-", "null", "NULL", "unknown", "Unknown", "UNKN", "UNK",
    " ", "", "ZZ", "XX", "EU",
]

# ── Dates: Italian month abbreviations ───────────────────────────────────────
ITALIAN_MONTHS = {
    "GEN": 1, "FEB": 2,  "MAR": 3,  "APR": 4,
    "MAG": 5, "GIU": 6,  "LUG": 7,  "AGO": 8,
    "SET": 9, "OTT": 10, "NOV": 11, "DIC": 12,
}

ITALIAN_MONTHS_LONG = {
    "GEN": "Jan", "FEB": "Feb", "MAR": "Mar", "APR": "Apr",
    "MAG": "May", "GIU": "Jun", "LUG": "Jul", "AGO": "Aug",
    "SET": "Sep", "OTT": "Oct", "NOV": "Nov", "DIC": "Dec",
}

# ── Year: known encoding corrections ─────────────────────────────────────────
ANNO_CORRECTIONS = {
    "24": "2024", "anno 2024": "2024",
    "2023": "2024",   # known encoding error in raw data (see EDA)
    "2024.": "2024", "2024": "2024",
}

# ── Country: ISO2 → ISO3 complete mapping ─────────────────────────────────────
ISO2_TO_ISO3 = {
    "IT": "ITA", "AL": "ALB", "TR": "TUR", "AE": "ARE", "GB": "GBR",
    "EG": "EGY", "DE": "DEU", "FR": "FRA", "ES": "ESP", "PT": "PRT",
    "NL": "NLD", "BE": "BEL", "CH": "CHE", "AT": "AUT", "GR": "GRC",
    "HR": "HRV", "RS": "SRB", "BG": "BGR", "RO": "ROU", "PL": "POL",
    "UA": "UKR", "RU": "RUS", "US": "USA", "CA": "CAN", "MX": "MEX",
    "BR": "BRA", "AR": "ARG", "CN": "CHN", "JP": "JPN", "KR": "KOR",
    "IN": "IND", "PK": "PAK", "BD": "BGD", "TH": "THA", "VN": "VNM",
    "ID": "IDN", "PH": "PHL", "MY": "MYS", "SG": "SGP", "AU": "AUS",
    "NZ": "NZL", "ZA": "ZAF", "NG": "NGA", "KE": "KEN", "ET": "ETH",
    "MA": "MAR", "TN": "TUN", "DZ": "DZA", "LY": "LBY", "SD": "SDN",
    "SA": "SAU", "QA": "QAT", "KW": "KWT", "IR": "IRN", "IQ": "IRQ",
    "SY": "SYR", "LB": "LBN", "JO": "JOR", "IL": "ISR", "AF": "AFG",
    "AO": "AGO", "AD": "AND", "MD": "MDA", "MK": "MKD", "XK": "RKS",
    "MV": "MDV", "AZ": "AZE", "GE": "GEO", "AM": "ARM", "BY": "BLR",
    "LT": "LTU", "LV": "LVA", "EE": "EST", "FI": "FIN", "SE": "SWE",
    "NO": "NOR", "DK": "DNK", "IS": "ISL", "IE": "IRL", "CZ": "CZE",
    "SK": "SVK", "HU": "HUN", "SI": "SVN", "LU": "LUX", "MT": "MLT",
    "CY": "CYP",
}

# ── Gender ────────────────────────────────────────────────────────────────────
FEMALE_VALS = {"f", "femmina", "female", "donna", "f."}
MALE_VALS   = {"m", "maschio", "male", "uomo", "m."}

# ── Age group ─────────────────────────────────────────────────────────────────
FASCIA_ETA_VALID = {"0-17", "18-30", "31-45", "46-60", "61+", "N.D."}
FASCIA_MAP = {
    "minore": "0-17",
    "adulto": "31-45",
    "101+":   "61+",
    "-5":     np.nan,
    "N/C":    np.nan,
}

# ── Zone ──────────────────────────────────────────────────────────────────────
ZONE_VALIDE = {"1", "2", "3", "4", "5", "6", "7", "8", "9"}

# ── Invalid OCCORRENZE values ─────────────────────────────────────────────────
OCC_INVALID = {"???", "N/C", "ALLARMATI"}

# ── Sparse column threshold ───────────────────────────────────────────────────
NULL_DROP_THRESHOLD = 0.50


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file with automatic separator detection."""
    for sep in [",", ";", "\t"]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python", dtype=str)
            if df.shape[1] > 1:
                print(f"  Loaded '{path.name}' with sep='{sep}' "
                      f"— {df.shape[0]} rows, {df.shape[1]} columns")
                return df
        except Exception:
            continue
    raise ValueError(f"Cannot load {path}")


def parse_date(val) -> pd.Timestamp:
    """
    Robust parser for DATA_PARTENZA.
    Handles Italian months ('FEB 13 2024'), mixed formats and fallbacks.
    """
    if pd.isna(val):
        return pd.NaT
    val = str(val).strip()
    for ita, eng in ITALIAN_MONTHS_LONG.items():
        val = val.replace(ita, eng)
    for fmt in (
        "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d",
        "%Y/%m/%d", "%d.%m.%Y", "%d-%m-%y", "%b %d %Y",
    ):
        try:
            return pd.to_datetime(val, format=fmt)
        except Exception:
            pass
    try:
        return pd.to_datetime(val, dayfirst=True)
    except Exception:
        return pd.NaT


def extract_number(val) -> float:
    """
    Extracts numeric value from noisy strings.
    Handles: '123 pax', '~45', '1,5', '20 voli', etc.
    """
    if pd.isna(val):
        return np.nan
    val = (str(val).strip()
           .replace(",", ".")
           .replace("~", "")
           .replace("pax", "")
           .replace("voli", "")
           .strip())
    try:
        return float(val)
    except Exception:
        return np.nan


def normalize_gender(val) -> str:
    """
    Normalizes GENERE → M / F / NaN.
    Conservative: '1'/'2' → NaN (may be noise, not a gender code).
    """
    if pd.isna(val):
        return np.nan
    v = str(val).strip().lower()
    if v in FEMALE_VALS:
        return "F"
    if v in MALE_VALS:
        return "M"
    return np.nan


def apply_iso2_to_iso3(series: pd.Series) -> pd.Series:
    """Converts ISO2 country codes to ISO3. Already-ISO3 values pass through unchanged."""
    return series.str.strip().str.upper().replace(ISO2_TO_ISO3)


def drop_sparse_columns(df: pd.DataFrame, threshold: float = NULL_DROP_THRESHOLD) -> pd.DataFrame:
    """Drops columns whose null percentage exceeds the threshold."""
    to_drop = [c for c in df.columns if df[c].isna().mean() > threshold]
    if to_drop:
        print(f"  Dropped columns (>{threshold*100:.0f}% null): {to_drop}")
    return df.drop(columns=to_drop)


# ══════════════════════════════════════════════════════════════════════════════
#  ALLARMI CLEANING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def clean_allarmi(df: pd.DataFrame) -> pd.DataFrame:
    print("\n── Cleaning ALLARMI ─────────────────────────────────────")

    # 1. Drop duplicate columns
    cols_to_drop = [c for c in COLS_DROP_ALLARMI if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"  Dropped {len(cols_to_drop)} duplicate columns")

    # 2. Replace all NULL variants with NaN
    df = df.replace(NULL_VALUES, np.nan)

    # 3. ANNO_PARTENZA
    df["ANNO_PARTENZA"] = df["ANNO_PARTENZA"].replace(ANNO_CORRECTIONS)
    df["ANNO_PARTENZA"] = pd.to_numeric(df["ANNO_PARTENZA"], errors="coerce")

    # 4. MESE_PARTENZA: Italian text → numeric
    df["MESE_PARTENZA"] = df["MESE_PARTENZA"].replace(ITALIAN_MONTHS)
    df["MESE_PARTENZA"] = pd.to_numeric(df["MESE_PARTENZA"], errors="coerce").astype("Int64")

    # 5. DATA_PARTENZA with robust parser
    n_before = df["DATA_PARTENZA"].notna().sum()
    df["DATA_PARTENZA"] = df["DATA_PARTENZA"].apply(parse_date)
    n_after = df["DATA_PARTENZA"].notna().sum()
    print(f"  DATA_PARTENZA parsed: {n_after}/{n_before} valid")

    # 6. Overwrite ANNO and MESE from DATA_PARTENZA (more reliable)
    mask = df["DATA_PARTENZA"].notna()
    df.loc[mask, "ANNO_PARTENZA"] = df.loc[mask, "DATA_PARTENZA"].dt.year
    df.loc[mask, "MESE_PARTENZA"] = df.loc[mask, "DATA_PARTENZA"].dt.month

    # 7. TOT: extract number, remove negatives, non-integers and placeholders >9999
    df["TOT"] = df["TOT"].apply(extract_number)
    df.loc[df["TOT"] < 0, "TOT"] = np.nan
    df.loc[df["TOT"] != df["TOT"].round(), "TOT"] = np.nan
    df.loc[df["TOT"] > 9999, "TOT"] = np.nan

    # 8. ZONA
    df["ZONA"] = df["ZONA"].astype(str).str.strip()
    df.loc[~df["ZONA"].isin(ZONE_VALIDE), "ZONA"] = np.nan
    df["ZONA"] = pd.to_numeric(df["ZONA"], errors="coerce").astype("Int64")

    # 9. OCCORRENZE: invalid values → NaN
    df["OCCORRENZE"] = df["OCCORRENZE"].replace({v: np.nan for v in OCC_INVALID})

    # 10. Airport codes: strip + uppercase (merge bug fix)
    for col in ["AREOPORTO_ARRIVO", "AREOPORTO_PARTENZA"]:
        if col in df.columns:
            df[col] = df[col].str.strip().str.upper()

    # 11. Country codes: ISO2 → ISO3 complete
    for col in ["CODICE_PAESE_ARR", "CODICE_PAESE_PART"]:
        if col in df.columns:
            df[col] = apply_iso2_to_iso3(df[col])

    # 12. Strip string columns
    for col in ["PAESE_ARR", "PAESE_PART", "MOTIVO_ALLARME"]:
        if col in df.columns:
            df[col] = df[col].str.strip()

    # 13. Temporal features derived from DATA_PARTENZA
    df["ora_partenza"]     = df["DATA_PARTENZA"].dt.hour
    df["giorno_settimana"] = df["DATA_PARTENZA"].dt.dayofweek
    df["mese"]             = df["DATA_PARTENZA"].dt.month

    # 14. Drop sparse columns (>50% null)
    df = drop_sparse_columns(df)

    print(f"  Final shape: {df.shape}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  VIAGGIATORI CLEANING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def clean_viaggiatori(df: pd.DataFrame) -> pd.DataFrame:
    print("\n── Cleaning TIPOLOGIA_VIAGGIATORE ───────────────────────")

    # 1. Drop duplicate columns
    cols_to_drop = [c for c in COLS_DROP_VIAGGIATORI if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"  Dropped {len(cols_to_drop)} duplicate columns")

    # 2. Replace all NULL variants with NaN
    df = df.replace(NULL_VALUES, np.nan)

    # 3. ANNO_PARTENZA
    df["ANNO_PARTENZA"] = df["ANNO_PARTENZA"].replace(ANNO_CORRECTIONS)
    df["ANNO_PARTENZA"] = pd.to_numeric(df["ANNO_PARTENZA"], errors="coerce")

    # 4. MESE_PARTENZA: Italian text → numeric
    df["MESE_PARTENZA"] = df["MESE_PARTENZA"].replace(ITALIAN_MONTHS)
    df["MESE_PARTENZA"] = pd.to_numeric(df["MESE_PARTENZA"], errors="coerce").astype("Int64")

    # 5. DATA_PARTENZA with robust parser
    n_before = df["DATA_PARTENZA"].notna().sum()
    df["DATA_PARTENZA"] = df["DATA_PARTENZA"].apply(parse_date)
    n_after = df["DATA_PARTENZA"].notna().sum()
    print(f"  DATA_PARTENZA parsed: {n_after}/{n_before} valid")

    # 6. Overwrite ANNO and MESE from DATA_PARTENZA
    mask = df["DATA_PARTENZA"].notna()
    df.loc[mask, "ANNO_PARTENZA"] = df.loc[mask, "DATA_PARTENZA"].dt.year
    df.loc[mask, "MESE_PARTENZA"] = df.loc[mask, "DATA_PARTENZA"].dt.month

    # 7. ENTRATI, INVESTIGATI, ALLARMATI: extract number, remove negatives
    for col in ["ENTRATI", "INVESTIGATI", "ALLARMATI"]:
        if col in df.columns:
            df[col] = df[col].apply(extract_number)
            df.loc[df[col] < 0, col] = np.nan

    # 8. Domain constraints: INVESTIGATI ≤ ENTRATI, ALLARMATI ≤ ENTRATI
    if all(c in df.columns for c in ["ENTRATI", "INVESTIGATI", "ALLARMATI"]):
        df.loc[df["INVESTIGATI"] > df["ENTRATI"], "INVESTIGATI"] = np.nan
        df.loc[df["ALLARMATI"]   > df["ENTRATI"], "ALLARMATI"]   = np.nan

    # 9. GENERE: conservative, '1'/'2' → NaN
    before = df["GENERE"].value_counts().nlargest(3).to_dict()
    df["GENERE"] = df["GENERE"].apply(normalize_gender)
    print(f"  GENERE: {before} → {df['GENERE'].value_counts(dropna=False).to_dict()}")

    # 10. TIPO_DOCUMENTO
    tipo_doc_valid = {"Passaporto", "Carta d'identità", "Visto", "Permesso di soggiorno"}
    df["TIPO_DOCUMENTO"] = df["TIPO_DOCUMENTO"].where(
        df["TIPO_DOCUMENTO"].isin(tipo_doc_valid), other=np.nan
    )

    # 11. FASCIA_ETA: recover text labels, invalidate the rest
    df["FASCIA_ETA"] = df["FASCIA_ETA"].replace(FASCIA_MAP)
    df["FASCIA_ETA"] = df["FASCIA_ETA"].where(
        df["FASCIA_ETA"].isin(FASCIA_ETA_VALID), other=np.nan
    )

    # 12. ZONA
    df["ZONA"] = df["ZONA"].astype(str).str.strip()
    df.loc[~df["ZONA"].isin(ZONE_VALIDE), "ZONA"] = np.nan
    df["ZONA"] = pd.to_numeric(df["ZONA"], errors="coerce").astype("Int64")

    # 13. Airport codes: strip + uppercase
    for col in ["AREOPORTO_ARRIVO", "AREOPORTO_PARTENZA"]:
        if col in df.columns:
            df[col] = df[col].str.strip().str.upper()

    # 14. Country codes: ISO2 → ISO3
    for col in ["CODICE_PAESE_ARR", "CODICE_PAESE_PART"]:
        if col in df.columns:
            df[col] = apply_iso2_to_iso3(df[col])

    # 15. NAZIONALITA: ISO2 → ISO3, then invalidate anything that is not 3 letters
    if "NAZIONALITA" in df.columns:
        df["NAZIONALITA"] = df["NAZIONALITA"].str.strip().str.upper().replace(ISO2_TO_ISO3)
        df.loc[df["NAZIONALITA"].str.len() != 3, "NAZIONALITA"] = np.nan

    # 16. Derived features
    for col in ["ENTRATI", "ALLARMATI", "INVESTIGATI"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["tasso_allarme"]     = np.where(df["ENTRATI"] > 0, df["ALLARMATI"]   / df["ENTRATI"], 0.0)
    df["tasso_investigati"] = np.where(df["ENTRATI"] > 0, df["INVESTIGATI"] / df["ENTRATI"], 0.0)

    # 17. Temporal features
    df["ora_partenza"]     = df["DATA_PARTENZA"].dt.hour
    df["giorno_settimana"] = df["DATA_PARTENZA"].dt.dayofweek
    df["mese"]             = df["DATA_PARTENZA"].dt.month

    # 18. Drop sparse columns
    df = drop_sparse_columns(df)

    print(f"  Final shape: {df.shape}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  MERGE
# ══════════════════════════════════════════════════════════════════════════════

def merge_datasets(df_allarmi: pd.DataFrame, df_viaggiatori: pd.DataFrame) -> pd.DataFrame:
    """
    Join between ALLARMI and TIPOLOGIA_VIAGGIATORE on:
    AREOPORTO_ARRIVO + AREOPORTO_PARTENZA + DATA_PARTENZA (date only).
    Left join: keeps all ALLARMI records.
    """
    print("\n── Merging datasets ─────────────────────────────────────")

    df_allarmi     = df_allarmi.copy()
    df_viaggiatori = df_viaggiatori.copy()

    df_allarmi["_data_key"]     = pd.to_datetime(df_allarmi["DATA_PARTENZA"]).dt.date
    df_viaggiatori["_data_key"] = pd.to_datetime(df_viaggiatori["DATA_PARTENZA"]).dt.date

    join_keys = ["AREOPORTO_ARRIVO", "AREOPORTO_PARTENZA", "_data_key"]

    agg_viaggiatori = df_viaggiatori.groupby(join_keys).agg(
        tot_entrati        = ("ENTRATI",          "sum"),
        tot_allarmati      = ("ALLARMATI",         "sum"),
        tot_investigati    = ("INVESTIGATI",       "sum"),
        tasso_allarme_volo = ("tasso_allarme",     "mean"),
        tasso_inv_volo     = ("tasso_investigati", "mean"),
        n_nazionalita      = ("NAZIONALITA",       "nunique"),
        n_respinti         = ("ESITO_CONTROLLO",   lambda x: (x == "RESPINTO").sum()),
        n_fermati          = ("ESITO_CONTROLLO",   lambda x: (x == "FERMATO").sum()),
        n_segnalati        = ("ESITO_CONTROLLO",   lambda x: (x == "SEGNALATO").sum()),
    ).reset_index()

    merged = df_allarmi.merge(agg_viaggiatori, on=join_keys, how="left")

    for _df in [merged, df_allarmi, df_viaggiatori]:
        if "_data_key" in _df.columns:
            _df.drop(columns=["_data_key"], inplace=True)

    print(f"  ALLARMI rows:      {len(df_allarmi)}")
    print(f"  VIAGGIATORI rows:  {len(df_viaggiatori)}")
    print(f"  Rows after merge:  {len(merged)}")
    print(f"  Matches found:     {merged['tot_entrati'].notna().sum()}/{len(merged)}")

    return merged


# ══════════════════════════════════════════════════════════════════════════════
#  QUALITY REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_quality_report(df: pd.DataFrame, name: str):
    print(f"\n{'='*55}")
    print(f"  Quality Report — {name}")
    print(f"{'='*55}")
    print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    if len(nulls) > 0:
        print("  Remaining null values:")
        for col, n in nulls.items():
            pct = n / len(df) * 100
            print(f"    {col:<35} {n:>5} ({pct:.1f}%)")
    else:
        print("  No significant nulls remaining.")
    print(f"{'='*55}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_preprocessing() -> tuple:
    """
    Runs the full preprocessing pipeline.
    Returns (df_allarmi_clean, df_viaggiatori_clean, df_merged).
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    df_allarmi     = load_csv(ALLARMI_PATH)
    df_viaggiatori = load_csv(VIAGGIATORI_PATH)

    df_allarmi     = clean_allarmi(df_allarmi)
    df_viaggiatori = clean_viaggiatori(df_viaggiatori)
    df_merged      = merge_datasets(df_allarmi, df_viaggiatori)

    print_quality_report(df_allarmi,     "ALLARMI clean")
    print_quality_report(df_viaggiatori, "VIAGGIATORI clean")
    print_quality_report(df_merged,      "MERGED")

    df_allarmi.to_csv(PROCESSED_DIR / "allarmi_clean.csv",         index=False)
    df_viaggiatori.to_csv(PROCESSED_DIR / "viaggiatori_clean.csv", index=False)
    df_merged.to_csv(PROCESSED_DIR / "dataset_merged.csv",         index=False)

    print(f"\nFiles saved to data/processed/")
    return df_allarmi, df_viaggiatori, df_merged


if __name__ == "__main__":
    run_preprocessing()
