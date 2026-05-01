"""Unit tests for the RiskProfilingAgent business rules.

These tests verify that:
  * Each of the five business rules fires at the exact threshold defined
    in BR_THRESHOLDS (and not before).
  * br_score is the simple mean of the five binary rules.
  * The blended confidence formula matches the documented weights.
  * final_risk classification follows the CRITICO/ALTO/MEDIO/BASSO ladder.

Run with::

    PYTHONPATH=. python -m pytest multiagent_pipeline/tests/test_risk_profiling_agent.py -v
"""
from __future__ import annotations

import pandas as pd
import pytest

from multiagent_pipeline.agents.risk_profiling_agent import (
    BR_THRESHOLDS,
    CONFIDENCE_WEIGHTS,
    _classify_final,
    run_risk_profiling_agent,
)


# ── Helpers ───────────────────────────────────────────────────────────────

def _make_state(rows: list[dict]) -> dict:
    """Builds the minimal state expected by run_risk_profiling_agent."""
    df = pd.DataFrame(rows)
    return {
        "df_anomalies": df,
        "anomaly_meta": {"n_alta": 0, "n_media": 0, "n_normale": len(df)},
    }


def _row(**overrides) -> dict:
    """Default 'NORMAL' row that fires no rules; tests override one field at a time."""
    base = {
        "ROTTA":               "TST-XXX",
        "PAESE_PART":          "Testland",
        "ZONA":                1,
        "risk_label":          "NORMALE",
        "ensemble_score":      0.0,
        "pct_interpol":        0.0,
        "pct_sdi":             0.0,
        "pct_nsis":            0.0,
        "tasso_respinti":      0.0,
        "tot_allarmi_log":     0.0,
        "tasso_chiusura":      1.0,   # high closure → low_closure rule must NOT fire
        "tasso_allarme_medio": 0.0,
    }
    base.update(overrides)
    return base


# ── Business-rule tests ──────────────────────────────────────────────────

def test_br_high_interpol_fires_at_threshold():
    """pct_interpol >= 0.30 → br_high_interpol = 1."""
    state = _make_state([
        _row(ROTTA="A", pct_interpol=0.29),                         # below threshold
        _row(ROTTA="B", pct_interpol=BR_THRESHOLDS["high_interpol_pct"]),  # exactly at
        _row(ROTTA="C", pct_interpol=0.50),                         # well above
    ])
    out = run_risk_profiling_agent(state)
    df = out["df_risk"].set_index("ROTTA")
    assert df.loc["A", "br_high_interpol"] == 0
    assert df.loc["B", "br_high_interpol"] == 1
    assert df.loc["C", "br_high_interpol"] == 1


def test_br_high_rejection_fires_at_threshold():
    """tasso_respinti >= 0.25 → br_high_rejection = 1."""
    state = _make_state([
        _row(ROTTA="A", tasso_respinti=0.24),
        _row(ROTTA="B", tasso_respinti=BR_THRESHOLDS["high_rejection_rate"]),
        _row(ROTTA="C", tasso_respinti=0.40),
    ])
    out = run_risk_profiling_agent(state)
    df = out["df_risk"].set_index("ROTTA")
    assert df.loc["A", "br_high_rejection"] == 0
    assert df.loc["B", "br_high_rejection"] == 1
    assert df.loc["C", "br_high_rejection"] == 1


def test_br_low_closure_requires_both_volume_and_low_rate():
    """br_low_closure fires only if tot_allarmi_log > 3 AND tasso_chiusura < 0.10."""
    state = _make_state([
        _row(ROTTA="A", tot_allarmi_log=2.0, tasso_chiusura=0.05),  # low volume → no
        _row(ROTTA="B", tot_allarmi_log=4.0, tasso_chiusura=0.20),  # high closure → no
        _row(ROTTA="C", tot_allarmi_log=4.0, tasso_chiusura=0.05),  # both met → yes
    ])
    out = run_risk_profiling_agent(state)
    df = out["df_risk"].set_index("ROTTA")
    assert df.loc["A", "br_low_closure"] == 0
    assert df.loc["B", "br_low_closure"] == 0
    assert df.loc["C", "br_low_closure"] == 1


def test_br_multi_source_requires_both_channels_active():
    """br_multi_source fires only when pct_interpol > 0 AND pct_sdi > 0."""
    state = _make_state([
        _row(ROTTA="A", pct_interpol=0.10, pct_sdi=0.00),  # only interpol
        _row(ROTTA="B", pct_interpol=0.00, pct_sdi=0.10),  # only sdi
        _row(ROTTA="C", pct_interpol=0.10, pct_sdi=0.10),  # both
    ])
    out = run_risk_profiling_agent(state)
    df = out["df_risk"].set_index("ROTTA")
    assert df.loc["A", "br_multi_source"] == 0
    assert df.loc["B", "br_multi_source"] == 0
    assert df.loc["C", "br_multi_source"] == 1


def test_br_high_alarm_rate_fires_at_threshold():
    """tasso_allarme_medio >= 0.50 → br_high_alarm_rate = 1."""
    state = _make_state([
        _row(ROTTA="A", tasso_allarme_medio=0.49),
        _row(ROTTA="B", tasso_allarme_medio=BR_THRESHOLDS["high_alarm_rate"]),
        _row(ROTTA="C", tasso_allarme_medio=0.90),
    ])
    out = run_risk_profiling_agent(state)
    df = out["df_risk"].set_index("ROTTA")
    assert df.loc["A", "br_high_alarm_rate"] == 0
    assert df.loc["B", "br_high_alarm_rate"] == 1
    assert df.loc["C", "br_high_alarm_rate"] == 1


# ── Aggregate tests ───────────────────────────────────────────────────────

def test_br_score_is_mean_of_five_rules():
    """br_score = sum(br_*) / 5 — verified at 0/5, 3/5, 5/5 hits."""
    state = _make_state([
        _row(ROTTA="zero"),                                       # 0 rules
        _row(                                                     # 3 rules
            ROTTA="three",
            pct_interpol=0.30,
            tasso_respinti=0.25,
            tasso_allarme_medio=0.50,
        ),
        _row(                                                     # 5 rules
            ROTTA="five",
            pct_interpol=0.40, pct_sdi=0.10,
            tasso_respinti=0.30,
            tot_allarmi_log=4.0, tasso_chiusura=0.05,
            tasso_allarme_medio=0.60,
        ),
    ])
    out = run_risk_profiling_agent(state)
    df = out["df_risk"].set_index("ROTTA")
    assert df.loc["zero", "br_score"]  == pytest.approx(0.0)
    assert df.loc["three", "br_score"] == pytest.approx(0.6)
    assert df.loc["five",  "br_score"] == pytest.approx(1.0)


def test_confidence_blends_ml_and_rules_with_documented_weights():
    """confidence = 0.6 * ensemble_score + 0.4 * br_score."""
    state = _make_state([
        _row(
            ROTTA="alpha",
            ensemble_score=0.50,
            pct_interpol=0.30,           # 1 rule fires → br_score = 0.2
        ),
    ])
    out = run_risk_profiling_agent(state)
    row = out["df_risk"].set_index("ROTTA").loc["alpha"]
    expected = (
        CONFIDENCE_WEIGHTS["ml"]    * 0.50
        + CONFIDENCE_WEIGHTS["rules"] * 0.20
    )
    assert row["confidence"] == pytest.approx(expected, abs=1e-4)


# ── Final-risk classification tests ───────────────────────────────────────

@pytest.mark.parametrize(
    "ml_label,br_score,expected",
    [
        ("ALTA",    0.6, "CRITICO"),  # ALTA + br_score >= 0.4 → CRITICO
        ("ALTA",    0.2, "ALTO"),     # ALTA but br_score < 0.4 → ALTO
        ("MEDIA",   0.6, "ALTO"),     # MEDIA + br_score >= 0.4 → ALTO
        ("MEDIA",   0.2, "MEDIO"),    # MEDIA, low rules → MEDIO
        ("NORMALE", 1.0, "BASSO"),    # NORMALE always BASSO regardless of rules
        ("NORMALE", 0.0, "BASSO"),
    ],
)
def test_classify_final_ladder(ml_label: str, br_score: float, expected: str):
    assert _classify_final(ml_label, br_score) == expected
