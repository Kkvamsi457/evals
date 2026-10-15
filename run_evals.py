"""Run Sarvam-105B evaluations and export JSON, CSV and HTML report."""
import argparse
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import ROOT_DIR, settings
from evaluators.metrics import (
    basic_code_switch_checks, reasoning_metrics, summarization_metrics, translation_metrics
)
from report_generator import generate_report
from sarvam_client import SarvamClient

def load_dataset(path: Path) -> list[dict]:
    if not path.is_absolute():
        path = ROOT_DIR / path
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    cases = []
    seen = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on dataset line {line_no}: {exc}") from exc
            for field in ("id", "language", "task", "prompt"):
                if not case.get(field):
                    raise ValueError(f"Dataset line {line_no} is missing required field '{field}'")
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

def evaluate_case(case: dict, client: SarvamClient, judge=None) -> dict:
    base = {
        "run_id": RUN_ID, "id": case["id"], "language": case["language"], "task": case["task"],
        "prompt": case["prompt"], "reference_answer": case.get("reference_answer"),
        "dataset_version": case.get("dataset_version", "starter-v1"),
        "model": settings.model, "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        generated = client.generate(case["prompt"])
        answer = generated["answer"]
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
        if judge is not None and task == "code_switching":
            try:
                judge_result = judge.evaluate(
                    language=case["language"], task=task, prompt=case["prompt"],
                    answer=answer, reference=reference
                )
                metrics["judge_correctness"] = judge_result.get("correctness")
                metrics["judge_relevance"] = judge_result.get("relevance")
                metrics["judge_language_fit"] = judge_result.get("language_fit")
            except Exception as exc:
                logging.exception("Judge failed for case %s", case["id"])
                judge_result = {"error": str(exc)}
        return {
            **base, "status": "success", "answer": answer, "metrics": metrics,
            "latency_seconds": generated["latency_seconds"], "usage": generated.get("usage", {}),
            "provider_model": generated.get("provider_model", settings.model),
            "judge": judge_result, "error": None,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        logging.exception("Evaluation failed for case %s", case["id"])
        return {
            **base, "status": "error", "answer": None, "metrics": {},
            "latency_seconds": None, "usage": {}, "provider_model": settings.model,
            "judge": None, "error": str(exc),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        }

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]

def main():
    parser = argparse.ArgumentParser(description="Evaluate Sarvam-105B across five Indian languages.")
    parser.add_argument("--dataset", default=str(settings.dataset_path), help="Path to JSONL evaluation dataset.")
    parser.add_argument("--output-dir", default=str(settings.output_dir), help="Directory for JSON, CSV, log and HTML outputs.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum cases to run; 0 means all.")
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

    if not settings.api_key or settings.api_key == "replace_with_your_sarvam_api_key":
        raise SystemExit("SARVAM_API_KEY is not set. Copy .env.example to .env and add your Sarvam API key.")

    judge = None
    if settings.judge_enabled:
        from judge import LLMJudge
        judge = LLMJudge(
            api_key=settings.judge_api_key,
            base_url=settings.judge_base_url,
            model=settings.judge_model,
            temperature=settings.judge_temperature,
        )

    client = SarvamClient(
        api_key=settings.api_key, model=settings.model, temperature=settings.temperature,
        max_tokens=settings.max_tokens, max_retries=settings.max_retries,
    )
    metadata = {
        "run_id": RUN_ID,
        "model": settings.model,
        "dataset_path": str(dataset_path.relative_to(ROOT_DIR)) if dataset_path.is_relative_to(ROOT_DIR) else str(dataset_path),
        "dataset_version": "starter-v1 (individual cases may override)",
        "cases_selected": len(cases),
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
        "judge_enabled": judge is not None,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
    }

    results = []
    for index, case in enumerate(cases, start=1):
        logging.info("[%s/%s] %s | %s | %s", index, len(cases), case["id"], case["language"], case["task"])
        result = evaluate_case(case, client, judge=judge)
        results.append(result)
        if result["status"] == "success":
            logging.info("  status=success latency=%s metrics=%s", result.get("latency_seconds"), result.get("metrics"))
        else:
            logging.error("  status=error error=%s", result.get("error"))
        if index < len(cases) and settings.request_delay_seconds > 0:
            time.sleep(settings.request_delay_seconds)

    metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["successful_calls"] = sum(r["status"] == "success" for r in results)
    metadata["failed_calls"] = sum(r["status"] != "success" for r in results)
    metadata["mean_successful_latency_seconds"] = (
        round(sum(r["latency_seconds"] for r in results if r["status"] == "success" and r["latency_seconds"] is not None) /
              sum(1 for r in results if r["status"] == "success" and r["latency_seconds"] is not None), 4)
        if any(r["status"] == "success" and r["latency_seconds"] is not None for r in results) else None
    )

    json_path = output_dir / "eval_results.json"
    csv_path = output_dir / "eval_results.csv"
    report_path = output_dir / "eval_report.html"
    json_path.write_text(json.dumps({"metadata": metadata, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")

    flat_rows = []
    for r in results:
        flat_rows.append({
            "run_id": r.get("run_id"), "id": r.get("id"), "language": r.get("language"),
            "task": r.get("task"), "status": r.get("status"), "model": r.get("model"),
            "latency_seconds": r.get("latency_seconds"), "answer": r.get("answer"),
            "metrics_json": json.dumps(r.get("metrics", {}), ensure_ascii=False),
            "usage_json": json.dumps(r.get("usage", {}), ensure_ascii=False),
            "judge_json": json.dumps(r.get("judge"), ensure_ascii=False),
            "error": r.get("error"), "started_at_utc": r.get("started_at_utc"),
            "finished_at_utc": r.get("finished_at_utc"),
        })
    pd.DataFrame(flat_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    generate_report(results, report_path, metadata)

    logging.info("Finished: %s successful, %s failed.", metadata["successful_calls"], metadata["failed_calls"])
    logging.info("JSON: %s", json_path)
    logging.info("CSV: %s", csv_path)
    logging.info("HTML: %s", report_path)
    print("\\nDone. Open results/eval_report.html in your browser.")
    if metadata["failed_calls"]:
        print("Some API calls failed. Review results/eval_results.json and results/execution.log.")

if __name__ == "__main__":
    main()
