from evaluators.metrics import (
    basic_code_switch_checks, exact_match_score, reasoning_metrics,
    summarization_metrics, translation_metrics
)

def test_normalized_exact_match():
    assert exact_match_score(" Mars! ", "mars") == 1.0

def test_reasoning_multiple_choice():
    result = reasoning_metrics("B. Mars", "B")
    assert result["correct"] is True
    assert result["exact_match"] == 1.0

def test_reasoning_wrong_choice():
    result = reasoning_metrics("A. Venus", "B")
    assert result["correct"] is False

def test_translation_metrics_exist():
    result = translation_metrics("नमस्ते दुनिया", "नमस्ते दुनिया")
    assert result["bleu"] is not None
    assert result["chrf"] is not None

def test_summary_metric_exists():
    result = summarization_metrics("Solar panels make electricity.", "Solar panels make electricity.")
    assert result["rougeL_f1"] == 100.0

def test_code_switch_smoke_check():
    result = basic_code_switch_checks("Photosynthesis plants food banane ka process hai.")
    assert result["non_empty"] is True
    assert result["word_count"] > 0
