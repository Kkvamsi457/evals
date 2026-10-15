# Sarvam-105B Multilingual Evaluation Framework

A Python evaluation harness for the task described in **Evals Task.pdf**. It evaluates Sarvam-105B across Hindi, Bengali, Telugu, Tamil and Marathi for:

1. Cross-lingual translation and summarization
2. Reasoning and knowledge questions
3. Conversational code-switching

It saves execution-level results as JSON/CSV and creates a self-contained `eval_report.html`.

> **Important:** The included dataset is a small starter/demo dataset to verify the pipeline. It is not a statistically representative benchmark. Expand and human-review the dataset before making broad claims about model quality. The project does not contain fabricated evaluation scores; scores are created only after you run it.

## Requirements

- Windows 10/11, macOS or Linux
- Python 3.11 recommended
- A Sarvam AI API key with access to `sarvam-105b`
- Internet access for live API calls

Official API docs: https://docs.sarvam.ai/api/api-guides-tutorials/chat-completion/overview

## Setup in VS Code (Windows PowerShell)

1. Extract the ZIP and open the `sarvam_multilingual_evals` folder in VS Code.
2. Open **Terminal → New Terminal**.
3. Create and activate a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this for the current terminal only, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

4. Copy `.env.example` to `.env` and paste your key into `SARVAM_API_KEY`:

```powershell
Copy-Item .env.example .env
```

Do not share `.env` or commit it to Git.

5. Run a small smoke test:

```powershell
python run_evals.py --limit 2
```

6. Run the full starter dataset:

```powershell
python run_evals.py
```

7. Open `results/eval_report.html` in your browser. The script also writes:
   - `results/eval_results.json`
   - `results/eval_results.csv`
   - `results/execution.log`
   - `results/eval_report.html`

## Dataset design

Dataset file: `datasets/eval_dataset.jsonl` (one JSON object per line).

Each case has:
- `id`: unique case identifier
- `language`: `hi`, `bn`, `te`, `ta`, or `mr`
- `task`: `translation`, `summarization`, `reasoning`, or `code_switching`
- `prompt`: task instructions and test input
- `reference_answer`: optional reference answer for supported reference-based metrics
- `source_text`: optional source text for ROUGE
- `metadata`: notes and benchmark provenance

The starter dataset includes 20 cases (four per language, one per task). This is only enough to verify execution, not to support statistically meaningful rankings.

## Metrics used

- **Translation:** SacreBLEU and chrF when a reference is supplied. These are overlap-based metrics; they do not prove semantic correctness.
- **Summarization:** ROUGE-L against a reference summary when available. Use human review or a validated judge for factuality and completeness.
- **Reasoning:** normalized exact match / multiple-choice correctness when a reference answer is supplied.
- **Code-switching:** deterministic output-presence checks by default. For meaningful semantic scoring, enable a separate judge or conduct human review.
- **Operational:** elapsed time, status, error message, model ID and usage metadata when available.

No single aggregate score is presented as a universal quality measure. Compare task and language scores separately and inspect individual failures.

## Optional LLM-as-a-Judge

The project can call a separate OpenAI-compatible judge endpoint. Configure these in `.env`:

```env
JUDGE_ENABLED=true
JUDGE_API_KEY=your_judge_provider_key
JUDGE_BASE_URL=https://your-openai-compatible-endpoint/v1
JUDGE_MODEL=your-approved-multilingual-judge-model
```

Use a model approved by your team and verify its support for all five languages. Do not silently use Sarvam-105B as its own independent judge. The judge output is advisory and must be validated against human-reviewed cases. API charges may apply.

The judge evaluates code-switching responses for intent, language/style fit, relevance and correctness. It returns a JSON score and rationale when possible. If the judge is disabled or fails, the main evaluation continues and the result records that no judge score is available.

## Reproducibility and interpretation

- Outputs may vary across requests even with temperature 0.
- API errors are recorded; failed cases are not silently counted as successful.
- Keep the same dataset version, model ID, generation settings and scoring code when comparing runs.
- Confirm that reference translations/summaries are reviewed by fluent speakers before treating the scores as benchmark results.
- If using translated MMLU/IndicQA, record source, translation process, answer key and any human validation in dataset metadata.
- Use a larger held-out dataset before drawing conclusions about language-level performance.

## Tests without an API key

The unit tests only check local scoring and report helpers; they do not call Sarvam.

```powershell
pytest -q
```

## CLI options

```powershell
python run_evals.py --help
python run_evals.py --limit 2
python run_evals.py --dataset datasets/eval_dataset.jsonl
python run_evals.py --output-dir results
```

`--limit 0` means all cases. `MAX_CASES` in `.env` is used if no `--limit` is passed.

## Project layout

```text
sarvam_multilingual_evals/
├── run_evals.py
├── sarvam_client.py
├── report_generator.py
├── config.py
├── evaluators/
│   ├── __init__.py
│   └── metrics.py
├── datasets/
│   └── eval_dataset.jsonl
├── results/
├── tests/
│   └── test_metrics.py
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```
