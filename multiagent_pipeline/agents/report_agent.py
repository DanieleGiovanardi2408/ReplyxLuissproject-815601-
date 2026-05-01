"""ReportAgent — fifth node of the multi-agent graph.

Responsibilities:
    Uses a real LLM to explain anomalous routes (ALTA/MEDIA) in natural
    language and build the final JSON report.
"""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
    __package__ = "multiagent_pipeline.agents"

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from langchain_anthropic import ChatAnthropic

from multiagent_pipeline.config import get_anthropic_api_key, get_anthropic_model
from multiagent_pipeline.state import AgentState, PATHS

logger = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Mapping z-score column → (absolute value column, human-readable label)
# These are the 11 baseline features computed by BaselineAgent.
_Z_COL_MAP: dict[str, tuple[str, str]] = {
    "z_pct_interpol":        ("pct_interpol",         "% allarmi Interpol"),
    "z_pct_sdi":             ("pct_sdi",              "% allarmi SDI"),
    "z_pct_nsis":            ("pct_nsis",             "% allarmi NSIS"),
    "z_tasso_rilevanza":     ("tasso_rilevanza",       "tasso rilevanza"),
    "z_tasso_inv_medio":     ("tasso_inv_medio",       "tasso investigazione"),
    "z_tasso_allarme_medio": ("tasso_allarme_medio",   "tasso allarme medio"),
    "z_tasso_chiusura":      ("tasso_chiusura",        "tasso chiusura"),
    "z_tasso_fermati":       ("tasso_fermati",         "tasso fermati"),
    "z_tasso_respinti":      ("tasso_respinti",        "tasso respinti"),
    "z_score_rischio_esiti": ("score_rischio_esiti",   "score rischio esiti"),
    "z_tot_allarmi_log":     ("tot_allarmi_sum",       "volume allarmi totali"),
}


def _fmt(v: object) -> str:
    """Formats a numeric value for the LLM prompt, handles NaN/None."""
    try:
        return f"{float(v):.3f}"  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "N/D"


def format_route_for_llm(row: dict) -> str:
    """Tool 1 — format_route_for_llm(row)

    Builds a structured context for the LLM by:
    1. Identifying the top-3 anomaly drivers (features with highest |z-score|
       relative to the historical baseline computed by BaselineAgent).
    2. Adding absolute values, individual model scores, and route context.

    This gives the LLM concrete, cited evidence instead of bare aggregate scores.
    """
    # ── Top-3 z-score drivers ──────────────────────────────────────────────
    z_scores: dict[str, float] = {}
    for z_col in _Z_COL_MAP:
        val = row.get(z_col)
        if val is not None:
            try:
                z_scores[z_col] = abs(float(val))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                pass

    top3 = sorted(z_scores, key=z_scores.__getitem__, reverse=True)[:3]

    driver_lines = []
    for z_col in top3:
        abs_col, label = _Z_COL_MAP[z_col]
        z_val = float(row.get(z_col, 0))  # type: ignore[arg-type]
        abs_val = row.get(abs_col)
        driver_lines.append(
            f"  - {label}: valore={_fmt(abs_val)}, deviazione={z_val:+.2f}σ dalla baseline"
        )

    drivers_str = (
        "\n".join(driver_lines) if driver_lines
        else "  - z-score non disponibili per questa rotta"
    )

    # ── Individual model scores ────────────────────────────────────────────
    models_str = (
        f"IsolationForest={_fmt(row.get('score_if'))}, "
        f"LOF={_fmt(row.get('score_lof'))}, "
        f"Autoencoder={_fmt(row.get('score_ae'))}"
    )

    # ── Route context ──────────────────────────────────────────────────────
    return (
        f"ROUTE: {row.get('ROTTA', 'N/A')} | "
        f"Departure country: {row.get('PAESE_PART', 'N/A')} | "
        f"Zone: {row.get('ZONA', 'N/A')}\n"
        f"Risk level: {row.get('risk_label', 'N/A')} | "
        f"Ensemble score: {_fmt(row.get('ensemble_score', 0))}\n"
        f"Score per model: {models_str}\n"
        f"Volumes: total_alarms={_fmt(row.get('tot_allarmi_sum'))}, "
        f"passengers_entered={_fmt(row.get('tot_entrati'))}\n"
        f"Top 3 anomaly drivers (features furthest from historical baseline):\n"
        f"{drivers_str}"
    )


def generate_explanation(context: str, llm: ChatAnthropic) -> str:
    """Tool 2 — generate_explanation(context)

    Generates a narrative explanation of the anomalous route using a real LLM.
    The prompt requires the LLM to integrate evidence from BOTH halves of
    the multi-agent pipeline: statistical drivers (z-scores produced by the
    BaselineAgent) and business-rule drivers (RiskProfilingAgent).
    """
    system = (
        "You are an airport security risk analyst writing for border-control "
        "operators. You MUST write your response in English only. "
        "Do not use Italian or any other language, regardless of the language of the input data."
    )
    user = (
        "Analyze the route data below and explain in 2-4 English sentences why this route is anomalous "
        "and what its risk profile means for operators.\n\n"
        "Your explanation MUST:\n"
        "  - cite at least TWO statistical drivers from the list, including each driver's numeric value "
        "and its deviation from the baseline (in sigma units);\n"
        "  - cite at least ONE business rule from the list of rules that fired, OR explicitly state "
        "that no business rule fired if the list is empty;\n"
        "  - reference the final risk classification (CRITICO / ALTO / MEDIO / BASSO) so the reader "
        "knows the operational severity, not just the ML score.\n\n"
        "Do not invent or assume any data that is not explicitly present below.\n\n"
        f"Route data:\n{context}\n\n"
        "IMPORTANT: Your answer must be in English only and must integrate both kinds of drivers."
    )
    from langchain_core.messages import SystemMessage, HumanMessage
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return response.content.strip()


def build_final_report(findings: list[dict], perimeter: dict, anomaly_meta: dict, n_tot: int) -> dict:
    """Tool 3 — build_final_report(findings)

    Builds the final JSON report with textual explanations.
    """
    n_alta = int(anomaly_meta.get("n_alta", 0))
    n_media = int(anomaly_meta.get("n_media", 0))
    n_normale = int(anomaly_meta.get("n_normale", 0))
    scope = ", ".join([f"{k}={v}" for k, v in perimeter.items()]) if perimeter else "no filter"
    summary = (
        f"Analysis completed on {n_tot} routes ({scope}). "
        f"Risk distribution: ALTA={n_alta}, MEDIA={n_media}, NORMALE={n_normale}. "
        f"LLM explanations were generated for {len(findings)} ALTA/MEDIA routes."
    )
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "perimeter": perimeter,
        "n_rotte_analizzate": n_tot,
        "distribution": {"alta": n_alta, "media": n_media, "normale": n_normale},
        "thresholds": {
            "soglia_alta": anomaly_meta.get("soglia_alta"),
            "soglia_media": anomaly_meta.get("soglia_media"),
        },
        "findings": findings,
        "summary": summary,
    }


def run_report_agent(
    state: AgentState,
    save_output: bool = True,
    output_path: Path | str | None = None,
    use_llm: bool = True,
    dry_run: bool = False,
) -> AgentState:
    """Generates the final JSON report using LLM and OutlierAgent output."""
    logger.info("ReportAgent -- Starting")

    try:
        if use_llm and not dry_run and not get_anthropic_api_key():
            raise ValueError("ANTHROPIC_API_KEY not set: cannot use the LLM ReportAgent.")

        # Prefer df_risk (contains business-rule columns, final_risk and
        # risk_drivers from the RiskProfilingAgent) and fall back to
        # df_anomalies for backward compatibility.
        df = state.get("df_risk")
        upstream_meta = state.get("risk_meta") or {}
        used_risk_layer = isinstance(df, pd.DataFrame) and not df.empty

        if not used_risk_layer:
            df = state.get("df_anomalies")
            upstream_meta = state.get("anomaly_meta") or {}
            logger.info("ReportAgent -- df_risk missing/empty, falling back to df_anomalies")

        if upstream_meta.get("error"):
            raise ValueError(f"upstream meta contains error: {upstream_meta['error']}")
        if df is None or not isinstance(df, pd.DataFrame):
            raise ValueError("df_risk/df_anomalies missing: run upstream agents first.")
        if df.empty:
            raise ValueError("upstream DataFrame is empty: cannot generate report.")
        # ``a_meta`` is what the existing summary builder expects (counts of
        # ALTA/MEDIA/NORMALE + thresholds). Always populate it from anomaly_meta
        # when present, falling back to the upstream meta otherwise.
        a_meta = state.get("anomaly_meta") or upstream_meta

        perimeter = state.get("perimeter") or {}
        n_tot = int(len(df))
        llm = None
        model_name = None
        llm_warning = None
        if use_llm and not dry_run:
            try:
                model_name = get_anthropic_model()
                llm = ChatAnthropic(model=model_name, temperature=0)
            except Exception as e:
                llm_warning = f"LLM init failed ({e}); using deterministic fallback."
                logger.warning("ReportAgent -- %s", llm_warning)

        # Routes to explain: all ALTA/MEDIA, sorted by descending score.
        # Full rows are passed to format_route_for_llm so it can extract
        # z-score drivers. The final findings dict only stores key fields.
        explain_df = (
            df[df["risk_label"].isin(["ALTA", "MEDIA"])]
            .sort_values("ensemble_score", ascending=False)
            .copy()
        )
        full_rows = explain_df.to_dict(orient="records")
        # Output columns include the RiskProfilingAgent layer so consumers
        # of the JSON report (Streamlit, downstream auditors) see the full
        # picture, not just the ML ensemble.
        _output_cols = ["ROTTA", "PAESE_PART", "ZONA", "risk_label",
                        "ensemble_score", "score_composito", "baseline_score",
                        "final_risk", "confidence", "br_score", "risk_drivers"]

        findings = []
        for row in full_rows:
            context = format_route_for_llm(row)
            if llm is None:
                explanation = (
                    "LLM explanation skipped (dry_run / use_llm=False). "
                    "See ensemble_score, baseline_score and z-score drivers in the dashboard."
                )
            else:
                try:
                    explanation = generate_explanation(context, llm)
                except Exception as e:
                    llm_warning = f"LLM call failed ({e}); local fallback activated."
                    logger.warning("ReportAgent -- %s", llm_warning)
                    explanation = (
                        "LLM explanation unavailable (API error). "
                        "Analyse the route using ensemble_score, baseline_score and z-score drivers."
                    )
            # Store only key fields in the output JSON (not all 65 columns)
            finding_record = {k: row.get(k) for k in _output_cols if k in row}
            finding_record["explanation"] = explanation
            findings.append(finding_record)

        report = build_final_report(
            findings=findings,
            perimeter=perimeter,
            anomaly_meta=a_meta,
            n_tot=n_tot,
        )
        if llm_warning:
            report["warning"] = llm_warning

        report_path = None
        if save_output:
            default_out = _PROJECT_ROOT / PATHS["multiagent_report"]
            out_path = Path(output_path) if output_path is not None else default_out
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
            report_path = str(out_path)
            logger.info("ReportAgent output saved to: %s", report_path)

        logger.info("ReportAgent ✓ Completed — routes=%d, explanations=%d", n_tot, len(findings))
        return {
            **state,
            "report": report,
            "report_path": report_path,
        }
    except Exception as e:
        logger.error("ReportAgent ✗ Error: %s", e)
        return {
            **state,
            "report": {"error": str(e), "user_message": "Report generation error: check configuration and inputs."},
            "report_path": None,
        }


if __name__ == "__main__":
    from multiagent_pipeline.agents.data_agent import data_agent_node
    
    from multiagent_pipeline.agents.baseline_agent import run_baseline_agent
    from multiagent_pipeline.agents.outlier_agent import run_outlier_agent

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    s: AgentState = {"perimeter": {"anno": 2024}}
    s = data_agent_node(s)
    s = run_baseline_agent(s)
    s = run_outlier_agent(s)
    s = run_report_agent(s)
    print("\n=== ReportAgent RESULT ===")
    print(s["report_path"])
    print((s["report"] or {}).get("summary"))
