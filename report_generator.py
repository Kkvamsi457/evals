"""Create a self-contained HTML report from evaluation results."""
import html
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

LANGUAGE_NAMES = {"hi": "Hindi", "bn": "Bengali", "te": "Telugu", "ta": "Tamil", "mr": "Marathi"}
TASK_NAMES = {
    "translation": "Translation", "summarization": "Summarization",
    "reasoning": "Reasoning & Knowledge", "code_switching": "Code-switching"
}

def _fmt(value, suffix=""):
    if value is None or value == "":
        return "N/A"
    try:
        return f"{float(value):.2f}{suffix}"
    except (TypeError, ValueError):
        return html.escape(str(value))

def generate_report(results: list[dict], out_path: Path, run_metadata: dict | None = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    successful = [r for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") != "success"]
    latencies = [r.get("latency_seconds") for r in successful if isinstance(r.get("latency_seconds"), (int, float))]
    mean_latency = sum(latencies) / len(latencies) if latencies else None

    grouped = defaultdict(list)
    for row in results:
        grouped[(row.get("language", "unknown"), row.get("task", "unknown"))].append(row)

    # CSS-only comparative charts for metrics that share a common scale.
    chart_specs = [
        ("bleu", "Translation BLEU", 100.0),
        ("chrf", "Translation chrF", 100.0),
        ("rougeL_f1", "Summarization ROUGE-L F1", 100.0),
        ("exact_match", "Reasoning exact match", 1.0),
        ("judge_language_fit", "Code-switching judge language fit", 5.0),
    ]
    chart_sections = []
    for metric_key, metric_title, scale_max in chart_specs:
        bars = []
        found = False
        for lang in ("hi", "bn", "te", "ta", "mr"):
            vals = []
            for row in successful:
                if row.get("language") != lang:
                    continue
                value = (row.get("metrics") or {}).get(metric_key)
                if isinstance(value, (int, float)) and value == value:
                    vals.append(float(value))
            if not vals:
                continue
            found = True
            avg = sum(vals) / len(vals)
            width = max(0.0, min(100.0, avg / scale_max * 100.0))
            bars.append(
                f"<div class='bar-row'><span>{LANGUAGE_NAMES.get(lang, lang)}</span>"
                f"<div class='bar-bg'><div class='bar' style='width:{width:.2f}%'></div></div>"
                f"<b>{avg:.2f}</b></div>"
            )
        if found:
            chart_sections.append(
                f"<div class='card chart-card'><h3>{html.escape(metric_title)}</h3>"
                f"<div class='label'>Mean score by language · scale maximum {scale_max:g}</div>"
                f"{''.join(bars)}</div>"
            )

    table_rows = []
    for lang in ("hi", "bn", "te", "ta", "mr"):
        for task in ("translation", "summarization", "reasoning", "code_switching"):
            rows = grouped.get((lang, task), [])
            if not rows:
                continue
            ok = [r for r in rows if r.get("status") == "success"]
            metric_names = []
            for r in ok:
                for k, v in (r.get("metrics") or {}).items():
                    if isinstance(v, (int, float)) and v == v:
                        metric_names.append(k)
            unique_metrics = list(dict.fromkeys(metric_names))
            metric_text = []
            for name in unique_metrics:
                vals = [r.get("metrics", {}).get(name) for r in ok]
                vals = [v for v in vals if isinstance(v, (int, float)) and v == v]
                if vals:
                    metric_text.append(f"{html.escape(name)}: {_fmt(sum(vals)/len(vals))}")
            table_rows.append(
                f"<tr><td>{LANGUAGE_NAMES.get(lang, lang)}</td>"
                f"<td>{TASK_NAMES.get(task, task)}</td><td>{len(rows)}</td>"
                f"<td>{len(ok)}</td><td>{len(rows)-len(ok)}</td>"
                f"<td>{' · '.join(metric_text) if metric_text else 'No numeric metric available'}</td></tr>"
            )

    detail_rows = []
    for r in results:
        answer = html.escape((r.get("answer") or r.get("error") or "")[:1400])
        metrics = html.escape(json.dumps(r.get("metrics", {}), ensure_ascii=False))
        detail_rows.append(
            "<tr>"
            f"<td>{html.escape(str(r.get('id','')))}</td>"
            f"<td>{html.escape(LANGUAGE_NAMES.get(r.get('language'), r.get('language','')))}</td>"
            f"<td>{html.escape(TASK_NAMES.get(r.get('task'), r.get('task','')))}</td>"
            f"<td><span class='status {html.escape(str(r.get('status','error')))}'>{html.escape(str(r.get('status','error')))}</span></td>"
            f"<td>{_fmt(r.get('latency_seconds'), ' s')}</td>"
            f"<td><details><summary>View output</summary><pre>{answer}</pre><p><b>Metrics:</b> {metrics}</p></details></td>"
            "</tr>"
        )

    metadata = run_metadata or {}
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    html_doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sarvam-105B Multilingual Evaluation Report</title>
<style>
:root{{--bg:#f4f7fb;--card:#fff;--ink:#172033;--muted:#64748b;--line:#dce4ef;--accent:#3859d6;}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}}
header{{background:linear-gradient(120deg,#14264e,#334fc2);color:#fff;padding:34px max(20px,calc((100% - 1200px)/2))}}
header h1{{margin:0 0 8px;font-size:clamp(25px,4vw,38px)}}header p{{margin:0;color:#dbe5ff}}
main{{max-width:1200px;margin:24px auto;padding:0 18px 40px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px;box-shadow:0 3px 12px #15284b08}}
.label{{color:var(--muted);font-size:13px}}.value{{font-size:29px;font-weight:750;margin-top:6px}}h2{{margin:28px 0 12px;font-size:22px}}
.table-wrap{{overflow-x:auto;background:#fff;border:1px solid var(--line);border-radius:12px}}table{{border-collapse:collapse;width:100%;min-width:780px}}th,td{{padding:12px 14px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}}th{{background:#eef2fa;font-size:13px}}tr:last-child td{{border-bottom:0}}
.status{{display:inline-block;padding:3px 8px;border-radius:20px;font-size:12px;font-weight:700}}.success{{background:#dcfce7;color:#166534}}.error{{background:#fee2e2;color:#991b1b}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;max-width:600px}}details summary{{cursor:pointer;color:var(--accent);font-weight:600}}
.note{{background:#fff7d6;border:1px solid #f0d981;padding:14px 16px;border-radius:10px;margin-top:18px;color:#614b0d}}
footer{{color:var(--muted);font-size:12px;margin-top:28px}}.bar-row{{display:grid;grid-template-columns:100px 1fr 45px;gap:10px;align-items:center;margin:10px 0}}.chart-card{{min-width:280px}}.chart-card h3{{margin:0 0 4px;font-size:16px}}.bar-bg{{background:#e4eaf5;height:12px;border-radius:8px;overflow:hidden}}.bar{{height:100%;background:var(--accent);width:0}}
@media(max-width:600px){{header{{padding:24px 18px}}main{{padding:0 12px 28px}}.value{{font-size:24px}}}}
</style></head><body>
<header><h1>Sarvam-105B Multilingual Evaluation</h1><p>Translation · Summarization · Reasoning & Knowledge · Conversational Code-switching</p></header>
<main>
<div class="note"><b>Interpretation note:</b> This report contains measured results from the current run. The starter dataset is a smoke-test dataset, not a representative benchmark. Scores should not be treated as definitive language rankings until the dataset and references are expanded and reviewed.</div>
<h2>Run summary</h2><section class="grid">
<div class="card"><div class="label">Test cases</div><div class="value">{len(results)}</div></div>
<div class="card"><div class="label">Successful calls</div><div class="value">{len(successful)}</div></div>
<div class="card"><div class="label">Failed calls</div><div class="value">{len(failed)}</div></div>
<div class="card"><div class="label">Mean successful-call latency</div><div class="value">{_fmt(mean_latency, ' s')}</div></div>
<div class="card"><div class="label">Model</div><div class="value" style="font-size:18px">{html.escape(str(metadata.get('model','sarvam-105b')))}</div></div>
<div class="card"><div class="label">Languages</div><div class="value">5</div></div>
</section>
<h2>Comparative charts</h2><section class="grid">{''.join(chart_sections) if chart_sections else '<div class="card">Charts appear after successful cases produce comparable numeric metrics.</div>'}</section>
<h2>Language and task comparison</h2><div class="table-wrap"><table><thead><tr><th>Language</th><th>Task</th><th>Cases</th><th>Successful</th><th>Failed</th><th>Available metrics (mean)</th></tr></thead><tbody>{''.join(table_rows) or '<tr><td colspan="6">No results yet. Run python run_evals.py.</td></tr>'}</tbody></table></div>
<h2>Individual test cases</h2><div class="table-wrap"><table><thead><tr><th>ID</th><th>Language</th><th>Task</th><th>Status</th><th>Latency</th><th>Output / details</th></tr></thead><tbody>{''.join(detail_rows) or '<tr><td colspan="6">No results yet.</td></tr>'}</tbody></table></div>
<h2>Run metadata</h2><div class="card"><pre>{html.escape(json.dumps(metadata, ensure_ascii=False, indent=2))}</pre></div>
<footer>Generated {generated}. Keep raw JSON/CSV alongside this report for auditability. Model outputs and automated judge scores can be wrong; review important cases manually.</footer>
</main></body></html>"""
    out_path.write_text(html_doc, encoding="utf-8")
