"""BaselineAgent — third node of the multi-agent graph.

Responsibilities:
    Computes a robust statistical baseline on per-route features
    and produces robust z-scores + baseline_score for the OutlierAgent node.
"""
from __future__ import annotations

# ── Bootstrap for direct execution ───────────────────────────────────────────
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

from multiagent_pipeline.state import AgentState, BASELINE_FEATURES

logger = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _robust_zscore(series: pd.Series) -> tuple[pd.Series, float, float]:
    """Returns robust z-score using median and MAD.

    Fallback to std-based z-score when MAD=0 (common for sparse features
    where median=0 but meaningful outliers exist above zero).
    This prevents z-score columns from being silently all-zero, which would
    make the BaselineAgent contribution invisible to the ReportAgent.
    """
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    median = float(s.median())
    mad = float((s - median).abs().median())
    if mad > 0:
        z = (s - median) / (1.4826 * mad)
        return z, median, mad
    # MAD=0: more than 50% of values equal the median (common for sparse features).
    # Fall back to std-based z-score so outliers above zero are still detected.
    std = float(s.std())
    if std == 0:
        return pd.Series(np.zeros(len(s)), index=s.index), median, 0.0
    z = (s - median) / std
    return z, median, std


def run_baseline_agent(
    state: AgentState,
    save_output: bool = False,
    output_path: Path | str | None = None,
) -> AgentState:
    """Computes robust baseline on `state['df_features']`."""
    logger.info("BaselineAgent ── Starting")
    started_at = time.perf_counter()

    try:
        df_features = state.get("df_features")
        if df_features is None or not isinstance(df_features, pd.DataFrame):
            raise ValueError("df_features missing: run DataAgent first.")
        if df_features.empty:
            raise ValueError("df_features is empty: cannot compute baseline.")

        feature_cols = [c for c in BASELINE_FEATURES if c in df_features.columns]
        missing_cols = [c for c in BASELINE_FEATURES if c not in df_features.columns]
        if not feature_cols:
            raise ValueError("No BASELINE_FEATURE available in df_features.")

        df_baseline = df_features.copy()
        stats = {}
        z_cols = []

        for col in feature_cols:
            z, med, mad = _robust_zscore(df_baseline[col])
            z_col = f"z_{col}"
            df_baseline[z_col] = z
            z_cols.append(z_col)
            stats[col] = {"median": round(med, 6), "mad": round(mad, 6)}

        # Baseline score: mean of the absolute robust deviations
        df_baseline["baseline_score"] = df_baseline[z_cols].abs().mean(axis=1)
        soglia_media = float(df_baseline["baseline_score"].quantile(0.90))
        soglia_alta = float(df_baseline["baseline_score"].quantile(0.97))

        df_baseline["baseline_flag"] = np.where(
            df_baseline["baseline_score"] >= soglia_alta,
            "ALTA",
            np.where(df_baseline["baseline_score"] >= soglia_media, "MEDIA", "NORMALE"),
        )

        saved_to = None
        if save_output:
            default_out = _PROJECT_ROOT / "data" / "processed" / "baseline_live.csv"
            out_path = Path(output_path) if output_path is not None else default_out
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df_baseline.to_csv(out_path, index=False)
            saved_to = str(out_path)
            logger.info("BaselineAgent output saved to: %s", saved_to)

        baseline_meta = {
            "n_features_baseline": len(feature_cols),
            "z_score_threshold": round(soglia_media, 6),
            "soglia_media": round(soglia_media, 6),
            "soglia_alta": round(soglia_alta, 6),
            "source": "computed_live",
            "n_rotte_con_zscore": int(len(df_baseline)),
            "feature_cols_used": feature_cols,
            "feature_cols_missing": missing_cols,
            "zscore_stats": stats,
            "saved_to": saved_to,
            "elapsed_s": round(time.perf_counter() - started_at, 3),
        }

        logger.info(
            "BaselineAgent ✓ Completed — routes=%d, features=%d",
            len(df_baseline),
            len(feature_cols),
        )
        return {
            **state,
            "df_baseline": df_baseline,
            "baseline_meta": baseline_meta,
        }
    except Exception as e:
        logger.error("BaselineAgent ✗ Error: %s", e)
        return {
            **state,
            "df_baseline": None,
            "baseline_meta": {
                "error": str(e),
                "user_message": "Baseline could not be computed: check that features are present and valid.",
                "elapsed_s": round(time.perf_counter() - started_at, 3),
            },
        }


if __name__ == "__main__":
    from multiagent_pipeline.agents.data_agent import data_agent_node
    from multiagent_pipeline.tools.data_tools import load_last_perimeter

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    _perimeter = load_last_perimeter() or {"anno": 2024}
    print(f"  Perimeter: {_perimeter}")
    state_in: AgentState = {"perimeter": _perimeter}
    with_features = data_agent_node(state_in)  # DataAgent now produces df_features inline
    out = run_baseline_agent(with_features)

    print("\n=== BaselineAgent RESULT ===")
    if out["baseline_meta"].get("error"):
        print("ERROR:", out["baseline_meta"]["error"])
    else:
        print("df_baseline shape:", out["df_baseline"].shape)
        print("n_features_baseline:", out["baseline_meta"]["n_features_baseline"])
        print("soglia_media:", out["baseline_meta"]["soglia_media"])
        print("soglia_alta:", out["baseline_meta"]["soglia_alta"])
