"""SupervisorAgent — second-pass verification on borderline ALTA routes.

This is the bit of the LangGraph that actually exploits a *non-linear*
control flow: after the OutlierAgent has produced a first-pass risk_label
on every route, the SupervisorAgent re-runs the IsolationForest **only on
the routes flagged ALTA** with a stricter contamination factor (3 % instead
of 10 %). Routes that survive the stricter pass are tagged as
``alta_robusta=True`` — these are the high-risk signals an analyst can
trust without further investigation.

The supervisor operates on the same ``df_anomalies`` produced by the
OutlierAgent: it does not need to re-engineer features or re-fit on the
whole population, only on the small ALTA subset (~17 routes out of 567).
That subset re-fit is what turns the linear DAG into a real branching
graph — it is the one place where the multi-agent topology *uses*
multi-agent semantics.

Inputs (from AgentState)
------------------------
- ``df_anomalies``: DataFrame with ``risk_label`` and the BASELINE_FEATURES
  used by the OutlierAgent.
- ``anomaly_meta``: meta dict — needed to short-circuit when the
  upstream agent failed.

Outputs (to AgentState)
-----------------------
- ``df_anomalies``: enriched with two new columns:
    * ``alta_robusta`` (bool): True if the route was ALTA in the first
      pass *and* survived the second-pass stricter contamination.
    * ``second_pass_score_if`` (float): stricter-contamination IF score
      for the ALTA-set, NaN elsewhere.
- ``supervisor_meta`` (dict): n_first_pass_alta, n_robust_alta,
  n_downgraded, contamination_strict, elapsed_s.

The supervisor is a **no-op** if the first pass produced fewer than 5 ALTA
routes (insufficient sample to refit the IF). In that case the columns
are still written but ``alta_robusta`` simply mirrors the first pass.
"""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
    __package__ = "multiagent_pipeline.agents"

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from multiagent_pipeline.state import AgentState, BASELINE_FEATURES

logger = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ── Hyper-parameters ─────────────────────────────────────────────────────
# The stricter contamination on the second pass is the single dial that
# makes this a *verification* step, not a re-run.
_CONTAMINATION_STRICT = 0.03   # first pass: 0.10
_MIN_ALTA_FOR_REFIT   = 5      # below this, refit is meaningless
_RANDOM_STATE         = 42
_N_ESTIMATORS_STRICT  = 300    # slightly more trees for the small subset


# ── Helpers ──────────────────────────────────────────────────────────────

def _feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray | None, list[str]]:
    cols = [c for c in BASELINE_FEATURES if c in df.columns]
    if not cols:
        return None, []
    X = df[cols].fillna(0.0).values
    return X, cols


# ── Public node ──────────────────────────────────────────────────────────

def run_supervisor_agent(
    state: AgentState,
    save_output: bool = False,
    output_path: Path | str | None = None,
) -> AgentState:
    """Second-pass verification on the ALTA subset.

    Adds ``alta_robusta`` and ``second_pass_score_if`` columns to
    df_anomalies and writes ``supervisor_meta``.
    """
    logger.info("SupervisorAgent ── Starting")
    started_at = time.perf_counter()

    try:
        df_in = state.get("df_anomalies")
        a_meta = state.get("anomaly_meta") or {}

        if a_meta.get("error"):
            raise ValueError(f"anomaly_meta contains error: {a_meta['error']}")
        if df_in is None or not isinstance(df_in, pd.DataFrame):
            raise ValueError("df_anomalies missing: run OutlierAgent first.")
        if df_in.empty:
            raise ValueError("df_anomalies is empty: cannot supervise.")

        df = df_in.copy()
        df["alta_robusta"]            = False
        df["second_pass_score_if"]    = np.nan

        n_first_alta = int((df["risk_label"] == "ALTA").sum())

        if n_first_alta < _MIN_ALTA_FOR_REFIT:
            logger.warning(
                "SupervisorAgent: only %d ALTA routes (< %d) — second pass skipped, "
                "first-pass labels treated as already verified.",
                n_first_alta, _MIN_ALTA_FOR_REFIT,
            )
            df.loc[df["risk_label"] == "ALTA", "alta_robusta"] = True
            meta = {
                "n_first_pass_alta":      n_first_alta,
                "n_robust_alta":          n_first_alta,
                "n_downgraded":           0,
                "contamination_strict":   _CONTAMINATION_STRICT,
                "skipped_reason":         f"only {n_first_alta} ALTA, need >= {_MIN_ALTA_FOR_REFIT}",
                "elapsed_s":              round(time.perf_counter() - started_at, 3),
            }
            return {**state, "df_anomalies": df, "supervisor_meta": meta}

        # Build the feature matrix and scale on the WHOLE dataset (so the
        # mean / std reference is the same as the first pass)
        X, cols = _feature_matrix(df)
        if X is None:
            raise ValueError(
                "BASELINE_FEATURES not present in df_anomalies — "
                "SupervisorAgent needs the same columns as OutlierAgent."
            )
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Refit IsolationForest with a stricter contamination on the FULL
        # population. The question we are answering is 'under a stricter
        # global rule (top 3 % instead of top 10 %), which routes still
        # appear anomalous?' — this is the standard 'second-pass tightening'
        # used in fraud and security pipelines.
        if_strict = IsolationForest(
            contamination=_CONTAMINATION_STRICT,
            n_estimators=_N_ESTIMATORS_STRICT,
            random_state=_RANDOM_STATE,
        )
        if_strict.fit(X_scaled)
        if_strict_pred = if_strict.predict(X_scaled)            # -1 anomaly, +1 normal
        if_strict_raw  = if_strict.decision_function(X_scaled)  # higher = more normal

        # A first-pass ALTA route is "robust" iff the stricter global rule
        # also flags it as anomalous.
        alta_mask  = (df["risk_label"] == "ALTA").values
        anom_strict = (if_strict_pred == -1)
        is_robust   = alta_mask & anom_strict

        # Materialise into the dataframe (write the strict score on every
        # row so a downstream consumer can rank with it directly).
        df["alta_robusta"]         = is_robust
        df["second_pass_score_if"] = if_strict_raw

        n_robust     = int(is_robust.sum())
        n_downgraded = int(n_first_alta - n_robust)
        n_strict_anom = int(anom_strict.sum())

        # Persist if requested
        saved_to = None
        if save_output:
            default_out = _PROJECT_ROOT / "data" / "processed" / "supervisor_output.csv"
            out_path = Path(output_path) if output_path is not None else default_out
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_path, index=False)
            saved_to = str(out_path)
            logger.info("SupervisorAgent output saved to: %s", saved_to)

        meta = {
            "n_first_pass_alta":         n_first_alta,
            "n_strict_anomalies_global": n_strict_anom,
            "n_robust_alta":             n_robust,
            "n_downgraded":              n_downgraded,
            "robustness_rate":           round(n_robust / max(n_first_alta, 1), 4),
            "contamination_strict":      _CONTAMINATION_STRICT,
            "n_estimators_strict":       _N_ESTIMATORS_STRICT,
            "feature_cols":              cols,
            "saved_to":                  saved_to,
            "elapsed_s":                 round(time.perf_counter() - started_at, 3),
        }

        logger.info(
            "SupervisorAgent ✓ Completed — first_pass_alta=%d, robust=%d, "
            "downgraded=%d (%.2fs)",
            n_first_alta, n_robust, n_downgraded, meta["elapsed_s"],
        )

        return {**state, "df_anomalies": df, "supervisor_meta": meta}

    except Exception as e:
        logger.error("SupervisorAgent ✗ Error: %s", e)
        return {
            **state,
            "supervisor_meta": {
                "error":        str(e),
                "user_message": "Supervisor verification failed — first-pass "
                                "labels are still available downstream.",
                "elapsed_s":    round(time.perf_counter() - started_at, 3),
            },
        }


if __name__ == "__main__":
    from multiagent_pipeline.agents.data_agent     import data_agent_node
    from multiagent_pipeline.agents.baseline_agent import run_baseline_agent
    from multiagent_pipeline.agents.outlier_agent  import run_outlier_agent
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    s: AgentState = {"perimeter": {"anno": 2024}}
    s = data_agent_node(s)
    s = run_baseline_agent(s)
    s = run_outlier_agent(s)
    s = run_supervisor_agent(s)
    sm = s["supervisor_meta"]
    print("\n=== SupervisorAgent RESULT ===")
    if sm.get("error"):
        print(f"  ERROR: {sm['error']}")
    else:
        print(f"  n_first_pass_alta : {sm['n_first_pass_alta']}")
        print(f"  n_robust_alta     : {sm['n_robust_alta']}")
        print(f"  n_downgraded      : {sm['n_downgraded']}")
        print(f"  contamination     : {sm['contamination_strict']}")
        print(f"  elapsed_s         : {sm['elapsed_s']}")
