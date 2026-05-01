"""End-to-end validation suite for the multi-agent pipeline.

Runs:
1) smoke test, no LLM (always)
2) base regression across 4 perimeters
3) optional LLM smoke on a small perimeter (enabled via RUN_LLM_SMOKE env var)
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from multiagent_pipeline.main import run_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "multiagent_validation_report.json"


def _run_case(
    name: str,
    perimeter: dict,
    *,
    run_report: bool,
    use_llm: bool,
    dry_run: bool,
) -> dict:
    _, summary = run_pipeline(
        perimeter=perimeter,
        run_report=run_report,
        use_llm=use_llm,
        dry_run=dry_run,
        continue_on_error=False,
        save_outputs=False,
    )
    ok = len(summary["failed_stages"]) == 0
    return {
        "name": name,
        "ok": ok,
        "perimeter": perimeter,
        "run_config": summary.get("run_config", {}),
        "runtime_s": summary.get("runtime_s"),
        "completed_stages": summary.get("completed_stages", []),
        "failed_stages": summary.get("failed_stages", []),
        "step_errors": summary.get("step_errors", {}),
        "report_path": summary.get("report_path"),
    }


def main() -> None:
    run_llm_smoke = os.getenv("RUN_LLM_SMOKE", "false").strip().lower() in {"1", "true", "yes", "y", "on"}
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_llm_smoke": run_llm_smoke,
        "results": [],
    }

    # 1) smoke, no LLM
    report["results"].append(
        _run_case(
            "smoke_no_llm",
            {"anno": 2024},
            run_report=True,
            use_llm=False,
            dry_run=True,
        )
    )

    # 2) base regression across 4 perimeters
    regression_cases = [
        ("reg_2024_all", {"anno": 2024}),
        ("reg_2024_algeria", {"anno": 2024, "paese_partenza": "Algeria"}),
        ("reg_2024_fco", {"anno": 2024, "aeroporto_arrivo": "FCO"}),
        ("reg_2024_zona1", {"anno": 2024, "zona": 1}),
    ]
    for name, perimeter in regression_cases:
        report["results"].append(
            _run_case(name, perimeter, run_report=False, use_llm=False, dry_run=True)
        )

    # 3) optional LLM smoke on a small perimeter
    if run_llm_smoke:
        report["results"].append(
            _run_case(
                "smoke_llm_small",
                {"anno": 2024, "paese_partenza": "Algeria"},
                run_report=True,
                use_llm=True,
                dry_run=False,
            )
        )

    ok_count = sum(1 for r in report["results"] if r["ok"])
    report["summary"] = {
        "n_tests": len(report["results"]),
        "n_ok": ok_count,
        "n_failed": len(report["results"]) - ok_count,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[E2E] report saved to: {OUT_PATH}")
    print(f"[E2E] summary: {report['summary']}")


if __name__ == "__main__":
    main()

