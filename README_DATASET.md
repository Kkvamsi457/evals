# Dataset notes

`eval_dataset.jsonl` has 20 starter cases: one translation, one summarization, one reasoning and one code-switching case for each of Hindi (`hi`), Bengali (`bn`), Telugu (`te`), Tamil (`ta`) and Marathi (`mr`).

This is a pipeline smoke-test dataset only. It is intentionally small and is not a substitute for a reviewed benchmark. The reference translations and summaries are starter references and have not been independently validated by fluent-speaker annotators. Before presenting results as a formal benchmark:
1. Have fluent speakers review each reference translation and summary.
2. Expand the number and diversity of cases per language and task.
3. Add normal, edge, ambiguous, adversarial and unanswerable cases.
4. Add translated MMLU or custom IndicQA questions with verified answer keys.
5. Record source/provenance and dataset version.
6. Keep a held-out set for final comparison.
