"""Multi-agent pipeline orchestrator (LangGraph).

Graph topology (5 spec-mandated agents + 1 verifier, 4 data-driven branches):

    START → DataAgent → BaselineAgent
                              │
              ┌─ baseline degenerate ─┴─ normal ─┐
              ▼                                  ▼
            (skip)                          OutlierAgent
              │                                  │
              │              ┌──── ALTA ≥ 5 ─────┴─── ALTA < 5 ───┐
              │              ▼                                    │
              │        SupervisorAgent                            │
              │              │                                    │
              │   ┌── downgrade > 50 % AND iter < cap ──┐         │
              │   ▼                                     ▼         │
              │   ↑─── (cycle back to OutlierAgent)     RiskProfilingAgent ◄┘
              │                                                   │
              └──────────────► RiskProfilingAgent ────────────────┘
                                                       │
                                          [ALTA/MEDIA present?]
                                                  ↓        ↓
                                                yes       no
                                                  ↓        ↓
                                             ReportAgent  END

Four real, data-driven conditional edges (separate from error-stop logic):

    1. after_baseline → outlier | risk
        Skips the heavy ML stack when the baseline signal is degenerate
        (n_features < 5 OR baseline_score std too low). The route then
        falls back to a pure rule-based path executed by the
        RiskProfilingAgent on raw features.
    2. after_outlier  → supervisor | risk
        Routes through the SupervisorAgent only when there are enough
        first-pass ALTA routes (≥ 5) to make a stricter refit
        statistically meaningful.
    3. after_supervisor → outlier | risk
        Cycles back to OutlierAgent when the SupervisorAgent disagrees
        with > 50 % of the first-pass ALTA labels — capped at
        _MAX_OUTLIER_ITERATIONS to guarantee termination.
    4. after_risk    → report | end
        Skips the LLM ReportAgent when there are no ALTA/MEDIA routes
        worth narrating (saves API cost on quiet perimeters).

DataAgent performs both perimeter filtering AND feature engineering (via
FeatureBuilder, the same module used by the classical pipeline) so the
visible agent count matches the spec's 5-agent topology. The
SupervisorAgent re-fits IsolationForest with a stricter contamination
(3 % instead of 10 %) and tags every first-pass ALTA route as
``alta_robusta=True`` only if it survives the tightened rule. The branch
is short-circuited when fewer than 5 ALTA routes are available —
refitting on a tiny subset would be statistically meaningless.

Each conditional edge also stops the graph on the first error unless
``continue_on_error=True``. Agents handle missing predecessor data
gracefully so partial diagnostics still surface to the UI.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from langgraph.graph import StateGraph, END

from multiagent_pipeline.agents.data_agent import data_agent_node
from multiagent_pipeline.agents.baseline_agent import run_baseline_agent
from multiagent_pipeline.agents.outlier_agent import run_outlier_agent
from multiagent_pipeline.agents.supervisor_agent import run_supervisor_agent
from multiagent_pipeline.agents.risk_profiling_agent import run_risk_profiling_agent
from multiagent_pipeline.agents.report_agent import run_report_agent
from multiagent_pipeline.config import get_dry_run, get_use_llm
from multiagent_pipeline.state import AgentState

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — branching thresholds
# ══════════════════════════════════════════════════════════════════════════════

# after_baseline: minimum baseline-score variance below which we consider the
# data "degenerate" (no real signal) and skip the ML ensemble entirely.
_MIN_BASELINE_STD = 0.01
# after_baseline: minimum number of baseline features available — below this,
# the ML models cannot be trained reliably so we route around them.
_MIN_BASELINE_FEATURES = 5
# after_supervisor: when more than this fraction of first-pass ALTA routes are
# downgraded to non-ALTA on the second pass, route the graph back to
# OutlierAgent to give the ensemble another shot with looser thresholds.
_DOWNGRADE_RETRY_RATE = 0.50
# Hard upper bound on OutlierAgent runs per pipeline invocation. Guarantees
# termination even when the supervisor keeps disagreeing.
_MAX_OUTLIER_ITERATIONS = 2


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _init_state(perimeter: dict) -> AgentState:
    return {
        "perimeter": perimeter or {},
        "df_raw": None,
        "df_allarmi": None,
        "df_viaggiatori": None,
        "data_meta": None,
        "df_features": None,
        "feature_meta": None,
        "df_baseline": None,
        "baseline_meta": None,
        "df_anomalies": None,
        "anomaly_meta": None,
        "supervisor_meta": None,
        "df_risk": None,
        "risk_meta": None,
        "report": None,
        "report_path": None,
        "outlier_iterations": 0,
    }


def _has_error(state: AgentState, meta_key: str) -> bool:
    """Checks whether a meta-dict contains an error."""
    meta = state.get(meta_key) or {}
    return isinstance(meta, dict) and bool(meta.get("error"))


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def _build_graph(
    *,
    save_outputs: bool,
    run_report: bool,
    use_llm: bool,
    dry_run: bool,
    continue_on_error: bool,
) -> Any:
    """Builds and compiles the LangGraph graph.

    Nodes wrap the existing agent functions, extracting only the fields
    that each agent writes to the state (delta update, not full state).

    Conditional edges implement the stop-on-error logic
    and the skip of ReportAgent when not needed.
    """

    # ── Nodes ─────────────────────────────────────────────────────────────────
    # Each node returns ONLY the keys it writes, not the full state.
    # LangGraph performs the merge automatically.

    def node_data(state: AgentState) -> dict:
        result = data_agent_node(state, save_artifacts=save_outputs)
        return {
            "df_raw":         result["df_raw"],
            "df_allarmi":     result["df_allarmi"],
            "df_viaggiatori": result["df_viaggiatori"],
            "data_meta":      result["data_meta"],
            "df_features":    result["df_features"],
            "feature_meta":   result["feature_meta"],
        }

    def node_baseline(state: AgentState) -> dict:
        result = run_baseline_agent(state, save_output=save_outputs)
        return {
            "df_baseline": result["df_baseline"],
            "baseline_meta": result["baseline_meta"],
        }

    def node_outlier(state: AgentState) -> dict:
        result = run_outlier_agent(state, save_output=save_outputs)
        # Track how many times OutlierAgent has run — used by the
        # `after_supervisor` cycle to enforce _MAX_OUTLIER_ITERATIONS.
        prev_iters = int(state.get("outlier_iterations") or 0)
        return {
            "df_anomalies": result["df_anomalies"],
            "anomaly_meta": result["anomaly_meta"],
            "outlier_iterations": prev_iters + 1,
        }

    def node_supervisor(state: AgentState) -> dict:
        # Second-pass IsolationForest with stricter contamination on the
        # ALTA subset. Adds alta_robusta + second_pass_score_if columns.
        result = run_supervisor_agent(state, save_output=save_outputs)
        return {
            "df_anomalies":    result["df_anomalies"],
            "supervisor_meta": result["supervisor_meta"],
        }

    def node_risk(state: AgentState) -> dict:
        result = run_risk_profiling_agent(state, save_output=save_outputs)
        return {
            "df_risk":   result["df_risk"],
            "risk_meta": result["risk_meta"],
        }

    def node_report(state: AgentState) -> dict:
        result = run_report_agent(
            state,
            save_output=save_outputs,
            use_llm=use_llm,
            dry_run=dry_run,
        )
        return {
            "report": result["report"],
            "report_path": result["report_path"],
        }

    # ── Conditional edges ────────────────────────────────────────────────────
    # If continue_on_error=True, always proceeds to the next node.
    # If False, stops at the first error.

    def after_data(state: AgentState) -> str:
        # DataAgent now also performs feature engineering, so we check both
        # data_meta and feature_meta for errors before moving on.
        if not continue_on_error and (
            _has_error(state, "data_meta") or _has_error(state, "feature_meta")
        ):
            return "end"
        return "baseline"

    def after_baseline(state: AgentState) -> str:
        if not continue_on_error and _has_error(state, "baseline_meta"):
            return "end"
        # Real data-driven branching: if the baseline signal is degenerate
        # (too few features available OR baseline_score has near-zero
        # variance — typical of perimeters that filter down to 1–2 routes
        # where every distance from the median collapses to zero), skip
        # the ML ensemble entirely. We terminate the graph here because a
        # rule-only RiskProfilingAgent without an `ensemble_score` would
        # misrepresent the result; the user gets a clear empty-output
        # signal with `baseline_meta` explaining why.
        meta = state.get("baseline_meta") or {}
        n_features = int(meta.get("n_features_baseline") or 0)
        df_baseline = state.get("df_baseline")
        baseline_std = 0.0
        n_rows = 0
        if df_baseline is not None and "baseline_score" in getattr(df_baseline, "columns", []):
            try:
                baseline_std = float(df_baseline["baseline_score"].std())
                n_rows = int(len(df_baseline))
            except Exception:
                baseline_std = 0.0
        if n_features < _MIN_BASELINE_FEATURES or baseline_std < _MIN_BASELINE_STD:
            logger.info(
                "Orchestrator -> baseline degenerate (n_features=%d, std=%.4f, "
                "n_rows=%d) — perimeter too narrow for ML detection, terminating",
                n_features, baseline_std, n_rows,
            )
            return "end"
        return "outlier"

    def after_outlier(state: AgentState) -> str:
        if not continue_on_error and _has_error(state, "anomaly_meta"):
            return "end"
        # Branch on the first-pass ALTA count. With < 5 ALTA we go straight
        # to the rule layer because a stricter refit on a tiny subset would
        # be statistically meaningless. With ≥ 5 we route through the
        # SupervisorAgent — this is the *real* branching point in the graph.
        df = state.get("df_anomalies")
        n_alta = 0
        if df is not None and hasattr(df, "columns") and "risk_label" in df.columns:
            n_alta = int((df["risk_label"] == "ALTA").sum())
        if n_alta >= 5:
            return "supervisor"
        return "risk"

    def after_supervisor(state: AgentState) -> str:
        if not continue_on_error and _has_error(state, "supervisor_meta"):
            return "end"
        # Real data-driven branching with cycle: if the SupervisorAgent
        # downgrades more than _DOWNGRADE_RETRY_RATE of the first-pass
        # ALTA labels, the OutlierAgent and the verifier disagree heavily —
        # signal that the contamination heuristic mis-calibrated the
        # threshold for this perimeter. Cycle back to OutlierAgent for a
        # second attempt (which will pick up `outlier_iterations` from
        # state and use it as a hint to widen contamination, see
        # OutlierAgent docstring). _MAX_OUTLIER_ITERATIONS bounds the
        # number of cycles to guarantee termination.
        sup_meta = state.get("supervisor_meta") or {}
        n_first_pass = int(sup_meta.get("n_first_pass_alta") or 0)
        n_downgraded = int(sup_meta.get("n_downgraded") or 0)
        downgrade_rate = (n_downgraded / n_first_pass) if n_first_pass > 0 else 0.0
        iters = int(state.get("outlier_iterations") or 0)
        if downgrade_rate > _DOWNGRADE_RETRY_RATE and iters < _MAX_OUTLIER_ITERATIONS:
            logger.info(
                "Orchestrator -> supervisor downgrade_rate=%.2f > %.2f and "
                "iter=%d < %d → cycling back to OutlierAgent",
                downgrade_rate, _DOWNGRADE_RETRY_RATE, iters, _MAX_OUTLIER_ITERATIONS,
            )
            return "outlier"
        return "risk"

    def after_risk(state: AgentState) -> str:
        if not continue_on_error and _has_error(state, "risk_meta"):
            return "end"
        if not run_report:
            return "end"
        # Skip report if there are no anomalous routes to explain.
        # Use df_risk (richer) when available, fall back to df_anomalies.
        df = state.get("df_risk")
        if df is None:
            df = state.get("df_anomalies")
        if df is not None and hasattr(df, "columns") and "risk_label" in df.columns:
            if not df["risk_label"].isin(["ALTA", "MEDIA"]).any():
                logger.info("Orchestrator -> no ALTA/MEDIA routes, skipping ReportAgent")
                return "end"
        elif df is None and not continue_on_error:
            return "end"
        return "report"

    # ── Graph assembly ───────────────────────────────────────────────────────

    graph = StateGraph(AgentState)

    graph.add_node("data", node_data)
    graph.add_node("baseline", node_baseline)
    graph.add_node("outlier", node_outlier)
    graph.add_node("supervisor", node_supervisor)
    graph.add_node("risk", node_risk)

    graph.set_entry_point("data")

    graph.add_conditional_edges(
        "data", after_data,
        {"baseline": "baseline", "end": END},
    )
    # Real branching: terminate early when the baseline signal is
    # degenerate (perimeter too narrow for ML — e.g. 1–2 routes left
    # after filtering). The user gets the partial state with the
    # baseline_meta explaining why no anomaly scores were produced.
    graph.add_conditional_edges(
        "baseline", after_baseline,
        {"outlier": "outlier", "end": END},
    )
    # Real branching: route through the SupervisorAgent only when we have
    # enough ALTA routes to make a stricter refit meaningful.
    graph.add_conditional_edges(
        "outlier", after_outlier,
        {"supervisor": "supervisor", "risk": "risk", "end": END},
    )
    # Real branching with cycle: feed the graph back to OutlierAgent when
    # the SupervisorAgent disagrees heavily (downgrade rate > 50 %), capped
    # by _MAX_OUTLIER_ITERATIONS so the cycle always terminates.
    graph.add_conditional_edges(
        "supervisor", after_supervisor,
        {"outlier": "outlier", "risk": "risk", "end": END},
    )

    if run_report:
        graph.add_node("report", node_report)
        graph.add_conditional_edges(
            "risk", after_risk,
            {"report": "report", "end": END},
        )
        graph.add_edge("report", END)
    else:
        graph.add_edge("risk", END)

    return graph.compile()


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY BUILDER
# ══════════════════════════════════════════════════════════════════════════════

_STAGE_META_KEYS = [
    ("data",       "data_meta"),
    ("baseline",   "baseline_meta"),
    ("outlier",    "anomaly_meta"),
    ("supervisor", "supervisor_meta"),
    ("risk",       "risk_meta"),
    ("report",     "report"),
]


def _build_summary(
    state: AgentState,
    started_at: float,
    run_config: dict,
) -> dict[str, Any]:
    """Builds the summary dict from the final state.

    Format identical to the previous version for compatibility with
    Streamlit and e2e tests.
    """
    stage_results: dict[str, dict[str, Any]] = {}
    step_errors: dict[str, str] = {}

    for stage_name, meta_key in _STAGE_META_KEYS:
        meta = state.get(meta_key)
        if meta is None:
            continue  # stage not executed
        err = meta.get("error") if isinstance(meta, dict) else None
        elapsed = meta.get("elapsed_s", 0) if isinstance(meta, dict) else 0
        stage_results[stage_name] = {
            "ok": err is None,
            "error": err,
            "elapsed_s": elapsed,
        }
        if err:
            step_errors[stage_name] = err

    return {
        "perimeter": state.get("perimeter"),
        "report_path": state.get("report_path"),
        "stages": stage_results,
        "step_errors": step_errors,
        "completed_stages": [k for k, v in stage_results.items() if v["ok"]],
        "failed_stages":    [k for k, v in stage_results.items() if not v["ok"]],
        "run_config": run_config,
        "runtime_s": round(time.perf_counter() - started_at, 3),
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    perimeter: dict | None = None,
    *,
    run_report: bool = False,
    use_llm: bool | None = None,
    dry_run: bool | None = None,
    continue_on_error: bool = False,
    save_outputs: bool = False,
) -> tuple[AgentState, dict[str, Any]]:
    """Runs the multi-agent pipeline as a LangGraph graph.

    API identical to the previous version for compatibility with
    Streamlit, script runner and e2e tests.

    Args:
        perimeter: user filters (anno, aeroporto, paese, zona).
        run_report: if True, also runs ReportAgent.
        use_llm: enables LLM calls in ReportAgent.
        dry_run: generates placeholder explanations without API calls.
        continue_on_error: if True, continues even after errors.
        save_outputs: saves CSV/JSON artefacts to disk.

    Returns:
        (final_state, summary) — same structure as the previous version.
    """
    use_llm_effective = get_use_llm(False) if use_llm is None else use_llm
    dry_run_effective = get_dry_run(False) if dry_run is None else dry_run

    run_config = {
        "run_report": run_report,
        "use_llm": use_llm_effective,
        "dry_run": dry_run_effective,
        "continue_on_error": continue_on_error,
        "save_outputs": save_outputs,
    }

    graph = _build_graph(
        save_outputs=save_outputs,
        run_report=run_report,
        use_llm=use_llm_effective,
        dry_run=dry_run_effective,
        continue_on_error=continue_on_error,
    )

    initial_state = _init_state(perimeter or {})
    started_at = time.perf_counter()

    logger.info(
        "Orchestrator LangGraph -> starting pipeline | perimeter=%s | config=%s",
        perimeter, run_config,
    )

    final_state = graph.invoke(initial_state)

    summary = _build_summary(final_state, started_at, run_config)

    logger.info(
        "Orchestrator LangGraph -> completed in %.2fs | stages=%s",
        summary["runtime_s"],
        {k: v["ok"] for k, v in summary["stages"].items()},
    )

    return final_state, summary


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    state, summary = run_pipeline(
        {"anno": 2024},
        run_report=False,
        save_outputs=False,
    )
    print("\n=== MULTIAGENT ORCHESTRATOR (LangGraph) ===")
    print(summary)
    if state.get("df_anomalies") is not None:
        print("df_anomalies shape:", state["df_anomalies"].shape)
