"""Run multilingual Sarvam evaluations and export JSON, CSV, and HTML reports."""

import argparse
import json
import logging
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import pandas as pd

from config import ROOT_DIR, settings
from evaluators.metrics import (
    basic_code_switch_checks,
    reasoning_metrics,
    summarization_metrics,
    translation_metrics,
)
from report_generator import generate_report
from sarvam_client import SarvamClient


RUN_ID = (
    datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    + "-"
    + uuid.uuid4().hex[:8]
)


def load_dataset(path: Path) -> list[dict]:
    """Load and validate a JSONL evaluation dataset."""
    if not path.is_absolute():
        path = ROOT_DIR / path

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    cases: list[dict] = []
    seen: set[str] = set()

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on dataset line {line_no}: {exc}"
                ) from exc

            for field in ("id", "language", "task", "prompt"):
                if not case.get(field):
                    raise ValueError(
                        f"Dataset line {line_no} is missing required field '{field}'"
                    )

            if case["id"] in seen:
                raise ValueError(f"Duplicate case id: {case['id']}")

            seen.add(case["id"])
            cases.append(case)

    return cases


def configure_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "execution.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


def _judge_score(judge_result: dict) -> float:
    """Aggregate the judge's three 1-5 rubric scores into a single score."""
    score_fields = ("correctness", "relevance", "language_fit")
    return round(mean(float(judge_result[field]) for field in score_fields), 2)


def evaluate_case(case: dict, client: SarvamClient, judge=None) -> dict:
    """Run the main model, task metrics, and optional LLM judge for one case."""
    base = {
        "run_id": RUN_ID,
        "id": case["id"],
        "language": case["language"],
        "task": case["task"],
        "prompt": case["prompt"],
        "reference_answer": case.get("reference_answer"),
        "dataset_version": case.get("dataset_version", "starter-v1"),
        "model": settings.model,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    try:
        generated = client.generate(case["prompt"])
        answer = generated.get("answer", "")
        if not isinstance(answer, str):
            answer = str(answer or "")

        task = case["task"]
        reference = case.get("reference_answer") or ""

        if task == "translation":
            metrics = translation_metrics(answer, reference)
        elif task == "summarization":
            metrics = summarization_metrics(answer, reference)
        elif task == "reasoning":
            metrics = reasoning_metrics(answer, reference)
        elif task == "code_switching":
            metrics = basic_code_switch_checks(answer)
        else:
            metrics = {}

        judge_result = None

        # Judge every task when enabled, not only code-switching cases.
        if judge is not None:
            try:
                judge_result = judge.evaluate(
                    language=case["language"],
                    task=task,
                    prompt=case["prompt"],
                    answer=answer,
                    reference=reference,
                )

                metrics["judge_correctness"] = judge_result["correctness"]
                metrics["judge_relevance"] = judge_result["relevance"]
                metrics["judge_language_fit"] = judge_result["language_fit"]
                metrics["judge_score"] = _judge_score(judge_result)
                metrics["judge_verdict"] = judge_result["verdict"]
                metrics["judge_reason"] = judge_result["rationale"]
                metrics["judge_unsupported_claims"] = judge_result["unsupported_claims"]

            except Exception as exc:
                # A judge failure should be recorded without marking the model call failed.
                logging.exception("Judge failed for case %s", case["id"])
                metrics["judge_score"] = None
                metrics["judge_verdict"] = None
                metrics["judge_reason"] = None
                metrics["judge_error"] = str(exc)
                judge_result = {"error": str(exc)}

        return {
            **base,
            "status": "success",
            "answer": answer,
            "metrics": metrics,
            "latency_seconds": generated.get("latency_seconds"),
            "usage": generated.get("usage", {}),
            "provider_model": generated.get("provider_model", settings.model),
            "judge": judge_result,
            "error": None,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as exc:
        logging.exception("Evaluation failed for case %s", case["id"])
        return {
            **base,
            "status": "error",
            "answer": None,
            "metrics": {},
            "latency_seconds": None,
            "usage": {},
            "provider_model": settings.model,
            "judge": None,
            "error": str(exc),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Sarvam across multilingual tasks."
    )
    parser.add_argument(
        "--dataset",
        default=str(settings.dataset_path),
        help="Path to JSONL evaluation dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(settings.output_dir),
        help="Directory for JSON, CSV, log, and HTML outputs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum cases to run; 0 means all.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = ROOT_DIR / dataset_path

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT_DIR / output_dir

    configure_logging(output_dir)

    cases = load_dataset(dataset_path)
    limit = settings.max_cases if args.limit is None else args.limit
    if limit and limit > 0:
        cases = cases[:limit]

    if not cases:
        raise SystemExit("No evaluation cases selected.")

    if (
        not settings.api_key
        or settings.api_key == "replace_with_your_sarvam_api_key"
    ):
        raise SystemExit(
            "SARVAM_API_KEY is not set. Add your Sarvam API key to .env."
        )

    judge = None
    if settings.judge_enabled:
        # Validate judge settings before spending money on model evaluation calls.
        from judge import LLMJudge

        judge = LLMJudge(
            api_key=settings.judge_api_key,
            base_url=settings.judge_base_url,
            model=settings.judge_model,
            temperature=settings.judge_temperature,
        )
        logging.info(
            "LLM judge enabled: model=%s; judging all selected tasks.",
            settings.judge_model,
        )
    else:
        logging.info("LLM judge disabled (JUDGE_ENABLED is not true).")

    client = SarvamClient(
        api_key=settings.api_key,
        model=settings.model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        max_retries=settings.max_retries,
    )

    metadata = {
        "run_id": RUN_ID,
        "model": settings.model,
        "dataset_path": (
            str(dataset_path.relative_to(ROOT_DIR))
            if dataset_path.is_relative_to(ROOT_DIR)
            else str(dataset_path)
        ),
        "dataset_version": "starter-v1 (individual cases may override)",
        "cases_selected": len(cases),
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
        "judge_enabled": judge is not None,
        "judge_model": settings.judge_model if judge is not None else None,
        "judge_scope": "all_tasks" if judge is not None else "disabled",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
    }

    results: list[dict] = []
    for index, case in enumerate(cases, start=1):
        logging.info(
            "[%s/%s] %s | %s | %s",
            index,
            len(cases),
            case["id"],
            case["language"],
            case["task"],
        )

        result = evaluate_case(case, client, judge=judge)
        results.append(result)

        if result["status"] == "success":
            logging.info(
                "  status=success latency=%s metrics=%s",
                result.get("latency_seconds"),
                result.get("metrics"),
            )
        else:
            logging.error(
                "  status=error error=%s",
                result.get("error"),
            )

        if index < len(cases) and settings.request_delay_seconds > 0:
            time.sleep(settings.request_delay_seconds)

    metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["successful_calls"] = sum(
        result["status"] == "success" for result in results
    )
    metadata["failed_calls"] = sum(
        result["status"] != "success" for result in results
    )

    successful_latencies = [
        result["latency_seconds"]
        for result in results
        if result["status"] == "success"
        and result["latency_seconds"] is not None
    ]
    metadata["mean_successful_latency_seconds"] = (
        round(mean(successful_latencies), 4) if successful_latencies else None
    )

    if judge is not None:
        judge_results = [
            result
            for result in results
            if result["status"] == "success" and result.get("judge") is not None
        ]
        judge_successes = [
            result
            for result in judge_results
            if "error" not in result["judge"]
        ]
        metadata["judge_cases_attempted"] = len(judge_results)
        metadata["judge_successful_cases"] = len(judge_successes)
        metadata["judge_failed_cases"] = len(judge_results) - len(judge_successes)
    else:
        metadata["judge_cases_attempted"] = 0
        metadata["judge_successful_cases"] = 0
        metadata["judge_failed_cases"] = 0

    json_path = output_dir / "eval_results.json"
    csv_path = output_dir / "eval_results.csv"
    report_path = output_dir / "eval_report.html"

    json_path.write_text(
        json.dumps(
            {"metadata": metadata, "results": results},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    flat_rows = []
    for result in results:
        flat_rows.append(
            {
                "run_id": result.get("run_id"),
                "id": result.get("id"),
                "language": result.get("language"),
                "task": result.get("task"),
                "status": result.get("status"),
                "model": result.get("model"),
                "latency_seconds": result.get("latency_seconds"),
                "answer": result.get("answer"),
                "metrics_json": json.dumps(
                    result.get("metrics", {}), ensure_ascii=False
                ),
                "usage_json": json.dumps(
                    result.get("usage", {}), ensure_ascii=False
                ),
                "judge_json": json.dumps(
                    result.get("judge"), ensure_ascii=False
                ),
                "error": result.get("error"),
                "started_at_utc": result.get("started_at_utc"),
                "finished_at_utc": result.get("finished_at_utc"),
            }
        )

    pd.DataFrame(flat_rows).to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )
    generate_report(results, report_path, metadata)

    logging.info(
        "Finished: %s successful, %s failed.",
        metadata["successful_calls"],
        metadata["failed_calls"],
    )
    logging.info("JSON: %s", json_path)
    logging.info("CSV: %s", csv_path)
    logging.info("HTML: %s", report_path)

    print("\nDone. Open results/eval_report.html in your browser.")
    if metadata["failed_calls"]:
        print(
            "Some model API calls failed. Review results/eval_results.json "
            "and results/execution.log."
        )
    if judge is not None and metadata["judge_failed_cases"]:
        print(
            "Some judge calls failed. Check judge_error in results/eval_results.json "
            "and the execution log."
        )


if __name__ == "__main__":
    main()
