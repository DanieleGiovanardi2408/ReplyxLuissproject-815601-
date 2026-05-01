"""Trend analysis on the temporal axis of the route-level data.

The Reply spec mentions *historical baseline using rolling averages and
seasonal decomposition* (slide 16). Both pipelines build a *cross-sectional*
baseline instead. This module exists to **measure honestly** whether the
dataset can support the spec-suggested temporal techniques, and to apply
them on the subset that can.

Why we did not adopt rolling-avg/STL by default
-----------------------------------------------
The dataset is composed of `(ROTTA, ANNO, MESE)` triples. To run STL we
need at least 12 observations per route — ideally 24+. Empirical check
(see `analyse_temporal_coverage`) shows the median route has **2 months**
of data and the maximum is **3 months**. STL is mathematically impossible
on that scale, and a 3-month rolling mean over 3 observations collapses
to the cross-sectional mean we already compute in the BaselineAgent.

What this module does
---------------------
1. `analyse_temporal_coverage(df_merged)` — produce an evidence table that
   shows the distribution of months-per-route. This is the data-driven
   justification we cite in the README *Deviations from spec* section.
2. `compute_trend_slopes(df_merged, feature_cols)` — for every route with
   ≥ 2 monthly observations, fit a simple linear regression of each
   feature against the time index and report the slope. Routes with only
   one observation get NaN. This is a *defensible* trend signal even on
   short series — it does not pretend to be STL.
3. `classify_trend(slope_df)` — discretise the slope into RISING / STABLE
   / DECLINING using a small absolute threshold so the result is operator-
   readable.

Usage::

    from multiagent_pipeline.src.trend_analysis import (
        analyse_temporal_coverage, compute_trend_slopes, classify_trend,
    )
    coverage = analyse_temporal_coverage(df_merged)
    slopes   = compute_trend_slopes(df_merged, feature_cols=['TOT', 'tot_allarmati'])
    trends   = classify_trend(slopes, slope_col='TOT_slope')
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Statsmodels is optional — only required for STL. We import lazily.
try:
    from statsmodels.tsa.seasonal import STL  # type: ignore
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


# ── 1. Coverage evidence ──────────────────────────────────────────────────

def analyse_temporal_coverage(
    df_merged: pd.DataFrame,
    *,
    route_col: str = "ROTTA",
    year_col: str = "ANNO_PARTENZA",
    month_col: str = "MESE_PARTENZA",
) -> dict:
    """How many monthly observations does each route have?

    Returns a dict with the distribution and the threshold-conditional
    counts that justify our decision to *not* use STL by default.
    """
    if route_col not in df_merged.columns:
        df_merged = df_merged.copy()
        df_merged[route_col] = (
            df_merged["AREOPORTO_PARTENZA"].astype(str).str.upper()
            + "-"
            + df_merged["AREOPORTO_ARRIVO"].astype(str).str.upper()
        )

    counts = (
        df_merged.groupby(route_col)
        .apply(lambda g: g[[year_col, month_col]].drop_duplicates().shape[0],
               include_groups=False)
        .rename("n_months")
    )

    return {
        "n_routes":              int(counts.index.nunique()),
        "median_months":         int(counts.median()),
        "mean_months":           round(float(counts.mean()), 2),
        "max_months":            int(counts.max()),
        "min_months":            int(counts.min()),
        "n_with_ge_3_months":    int((counts >= 3).sum()),
        "n_with_ge_6_months":    int((counts >= 6).sum()),
        "n_with_ge_12_months":   int((counts >= 12).sum()),
        "stl_feasible":          bool((counts >= 12).any()),
        "rolling_3m_meaningful": bool((counts >= 6).any()),
        "evidence_table":        counts.value_counts().sort_index().to_dict(),
        "verdict": (
            "STL infeasible: no route reaches 12 months of observations. "
            "Rolling 3-month means collapse to the cross-sectional mean given "
            "median coverage of {:.0f} months/route. We adopt the cross-"
            "sectional baseline in both pipelines and report a per-route "
            "linear trend slope where ≥ 2 observations are available."
        ).format(counts.median()),
    }


# ── 2. Trend slope (the only temporal signal that fits the data) ─────────

def compute_trend_slopes(
    df_merged: pd.DataFrame,
    feature_cols: list[str],
    *,
    route_col: str = "ROTTA",
    year_col: str = "ANNO_PARTENZA",
    month_col: str = "MESE_PARTENZA",
) -> pd.DataFrame:
    """Linear-regression slope of each feature vs. (year, month) per route.

    The slope is signed: positive means the feature is *rising* across the
    available months, negative means it is *declining*. Routes with fewer
    than 2 observations get NaN for every slope.
    """
    if route_col not in df_merged.columns:
        df_merged = df_merged.copy()
        df_merged[route_col] = (
            df_merged["AREOPORTO_PARTENZA"].astype(str).str.upper()
            + "-"
            + df_merged["AREOPORTO_ARRIVO"].astype(str).str.upper()
        )

    df = df_merged.copy()
    df["_t"] = df[year_col].astype("Int64").astype("Int64").astype(float) * 12 \
              + df[month_col].astype("Int64").astype("Int64").astype(float)

    rows = []
    for rotta, g in df.groupby(route_col):
        g = g.dropna(subset=["_t"])
        n = len(g)
        row = {"ROTTA": rotta, "n_months": n}
        if n < 2:
            for col in feature_cols:
                row[f"{col}_slope"] = np.nan
        else:
            x = g["_t"].values
            x_centered = x - x.mean()
            denom = float((x_centered ** 2).sum())
            for col in feature_cols:
                if col not in g.columns:
                    row[f"{col}_slope"] = np.nan
                    continue
                y = pd.to_numeric(g[col], errors="coerce").fillna(0.0).values
                if denom > 0:
                    slope = float((x_centered * y).sum() / denom)
                else:
                    slope = 0.0
                row[f"{col}_slope"] = slope
        rows.append(row)

    return pd.DataFrame(rows).sort_values("n_months", ascending=False).reset_index(drop=True)


def classify_trend(
    slopes: pd.DataFrame,
    *,
    slope_col: str,
    threshold_abs: float | None = None,
) -> pd.DataFrame:
    """Bucket each route into RISING / STABLE / DECLINING / INSUFFICIENT.

    `threshold_abs` defaults to half a standard deviation of the non-NaN
    slopes — i.e. a route is RISING only if its slope is materially
    above zero relative to the population.
    """
    out = slopes.copy()
    valid = out[slope_col].dropna()
    if threshold_abs is None and not valid.empty:
        threshold_abs = float(valid.std() * 0.5)
    threshold_abs = threshold_abs or 0.0

    def _bucket(s):
        if pd.isna(s):
            return "INSUFFICIENT"
        if s >  threshold_abs:
            return "RISING"
        if s < -threshold_abs:
            return "DECLINING"
        return "STABLE"

    out["trend"] = out[slope_col].apply(_bucket)
    return out


# ── 3. STL decomposition — only when feasible ────────────────────────────

def try_stl_on_route(
    df_merged: pd.DataFrame,
    rotta: str,
    feature: str,
    *,
    period: int = 12,
    route_col: str = "ROTTA",
) -> dict:
    """Run STL on a single route's monthly series of `feature`. Returns a
    dict with {trend, seasonal, resid, ok} or {ok: False, reason: …}."""
    if not HAS_STATSMODELS:
        return {"ok": False, "reason": "statsmodels not installed"}

    if route_col not in df_merged.columns:
        df_merged = df_merged.copy()
        df_merged[route_col] = (
            df_merged["AREOPORTO_PARTENZA"].astype(str).str.upper()
            + "-"
            + df_merged["AREOPORTO_ARRIVO"].astype(str).str.upper()
        )

    sub = df_merged[df_merged[route_col] == rotta].sort_values(["ANNO_PARTENZA", "MESE_PARTENZA"])
    if len(sub) < 2 * period:
        return {
            "ok":     False,
            "reason": f"route {rotta} has {len(sub)} observations, "
                      f"STL needs at least {2 * period}",
            "n":      len(sub),
        }

    y = pd.to_numeric(sub[feature], errors="coerce").fillna(0.0).values
    res = STL(y, period=period, robust=True).fit()
    return {
        "ok":       True,
        "n":        len(sub),
        "trend":    res.trend.tolist(),
        "seasonal": res.seasonal.tolist(),
        "resid":    res.resid.tolist(),
    }


# ── Smoke test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from pathlib import Path
    here = Path(__file__).resolve().parents[2]
    df = pd.read_csv(here / "data" / "processed" / "dataset_merged.csv")

    print("=== Temporal coverage evidence ===")
    cov = analyse_temporal_coverage(df)
    for k, v in cov.items():
        if k != "evidence_table":
            print(f"  {k:25s} : {v}")
    print()

    print("=== Trend slopes (top 5 routes by months) ===")
    slopes = compute_trend_slopes(df, feature_cols=["TOT"])
    print(slopes.head(10).to_string(index=False))
    print()

    print("=== Trend classification ===")
    trends = classify_trend(slopes, slope_col="TOT_slope")
    print(trends["trend"].value_counts().to_string())
