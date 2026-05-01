"""Threshold sensitivity analysis for the five business rules.

The five thresholds in `RiskProfilingAgent.BR_THRESHOLDS` are inherited
from the classical post-processing layer and were never perturbed: a
reviewer is right to ask *how much do the final ALTA / MEDIA / NORMALE
counts move if you shift one threshold by ±5 %?*

This module answers that question deterministically. For each threshold
in turn we:
    1. perturb its value by a list of relative deltas (e.g. ±5 %, ±10 %)
    2. recompute the five business rules → br_score → final_risk
    3. count how many routes land in each final-risk bucket

The output is a tidy DataFrame the caller can pivot into a heat-map.

Usage::

    from multiagent_pipeline.src.threshold_sensitivity import (
        run_sensitivity_analysis, plot_sensitivity_heatmap,
    )
    df_sens = run_sensitivity_analysis(df_anomalies)
    plot_sensitivity_heatmap(df_sens)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from multiagent_pipeline.agents.risk_profiling_agent import (
    BR_THRESHOLDS,
    CONFIDENCE_WEIGHTS,
    _classify_final,
    _safe_col,
)


# ── Default perturbation grid ─────────────────────────────────────────────
DEFAULT_DELTAS = [-0.10, -0.05, 0.0, +0.05, +0.10]
RISK_LEVELS    = ["CRITICO", "ALTO", "MEDIO", "BASSO"]


# ── Core: compute final_risk under a perturbed threshold dict ─────────────

def _compute_final_risk(df: pd.DataFrame, thresholds: dict) -> pd.Series:
    """Evaluates the five business rules on ``df`` with the supplied
    thresholds and returns the resulting `final_risk` Series."""
    pct_interpol         = _safe_col(df, "pct_interpol")
    pct_sdi              = _safe_col(df, "pct_sdi")
    tasso_respinti       = _safe_col(df, "tasso_respinti")
    tot_allarmi_log      = _safe_col(df, "tot_allarmi_log")
    tasso_chiusura       = _safe_col(df, "tasso_chiusura")
    tasso_allarme_medio  = _safe_col(df, "tasso_allarme_medio")

    br_high_interpol   = (pct_interpol         >= thresholds["high_interpol_pct"]).astype(int)
    br_high_rejection  = (tasso_respinti       >= thresholds["high_rejection_rate"]).astype(int)
    br_low_closure     = (
        (tot_allarmi_log >  thresholds["low_closure_volume"]) &
        (tasso_chiusura  <  thresholds["low_closure_rate"])
    ).astype(int)
    br_multi_source    = ((pct_interpol > 0) & (pct_sdi > 0)).astype(int)
    br_high_alarm_rate = (tasso_allarme_medio  >= thresholds["high_alarm_rate"]).astype(int)

    br_score = (
        br_high_interpol + br_high_rejection + br_low_closure
        + br_multi_source + br_high_alarm_rate
    ) / 5.0

    ensemble = pd.to_numeric(df.get("ensemble_score", 0), errors="coerce").fillna(0.0)
    confidence = (
        CONFIDENCE_WEIGHTS["ml"]    * ensemble
        + CONFIDENCE_WEIGHTS["rules"] * br_score
    )

    return pd.Series(
        [_classify_final(label, score)
         for label, score in zip(df.get("risk_label", "NORMALE"), br_score)],
        index=df.index,
    )


# ── Sensitivity loop ──────────────────────────────────────────────────────

def run_sensitivity_analysis(
    df: pd.DataFrame,
    *,
    thresholds: dict | None = None,
    deltas: list[float] | None = None,
) -> pd.DataFrame:
    """For every (threshold, delta) pair, recompute final_risk and report
    the count of routes per risk level.

    Parameters
    ----------
    df : pd.DataFrame
        Output of OutlierAgent (or df_anomalies). Must contain
        ``risk_label``, ``ensemble_score`` and the BASELINE_FEATURES.
    thresholds : dict, optional
        Baseline thresholds. Defaults to BR_THRESHOLDS.
    deltas : list[float], optional
        Relative perturbations. Defaults to ±10 %, ±5 %, 0 %.

    Returns
    -------
    pd.DataFrame
        Long-format table with columns
        ``threshold, delta_pct, perturbed_value, CRITICO, ALTO, MEDIO, BASSO``.
    """
    base = dict(thresholds or BR_THRESHOLDS)
    deltas = deltas or DEFAULT_DELTAS

    rows = []
    for tname, tvalue in base.items():
        for d in deltas:
            new_value = tvalue * (1 + d)
            perturbed = {**base, tname: new_value}
            final = _compute_final_risk(df, perturbed)
            counts = final.value_counts().reindex(RISK_LEVELS, fill_value=0)
            rows.append({
                "threshold":       tname,
                "baseline_value":  tvalue,
                "delta_pct":       d,
                "perturbed_value": new_value,
                **{lvl: int(counts[lvl]) for lvl in RISK_LEVELS},
            })

    return pd.DataFrame(rows)


# ── Pivot helper for the heat-map ─────────────────────────────────────────

def to_heatmap_matrix(
    df_sens: pd.DataFrame,
    *,
    risk_level: str = "ALTO",
) -> pd.DataFrame:
    """Pivots the long-format sensitivity table into (threshold × delta) for
    a single risk level. Useful for `seaborn.heatmap` rendering."""
    if risk_level not in RISK_LEVELS:
        raise ValueError(f"risk_level must be one of {RISK_LEVELS}")
    return df_sens.pivot(
        index="threshold",
        columns="delta_pct",
        values=risk_level,
    )


def summarise(df_sens: pd.DataFrame) -> pd.DataFrame:
    """Per-threshold summary: max absolute change in CRITICO+ALTO count
    when the threshold is perturbed across the delta grid. Quantifies
    which rule the system is *most* sensitive to."""
    df_sens = df_sens.copy()
    df_sens["high_risk"] = df_sens["CRITICO"] + df_sens["ALTO"]
    out = []
    for tname, group in df_sens.groupby("threshold"):
        baseline = group.loc[group["delta_pct"] == 0, "high_risk"].iloc[0]
        max_delta = (group["high_risk"] - baseline).abs().max()
        out.append({
            "threshold":              tname,
            "baseline_high_risk":     int(baseline),
            "max_abs_swing":          int(max_delta),
            "swing_pct_of_baseline":  round(100 * max_delta / max(baseline, 1), 1),
        })
    return pd.DataFrame(out).sort_values("max_abs_swing", ascending=False)


if __name__ == "__main__":
    from pathlib import Path
    here = Path(__file__).resolve().parents[2]
    df = pd.read_csv(here / "data" / "processed" / "anomaly_results_live.csv")

    sens = run_sensitivity_analysis(df)
    print("=== Long-format sensitivity table ===")
    print(sens.to_string(index=False))
    print()
    print("=== Per-threshold sensitivity summary ===")
    print(summarise(sens).to_string(index=False))
