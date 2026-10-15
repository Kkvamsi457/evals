"""Task-specific evaluation metrics with Unicode-aware text processing."""

import re
import unicodedata
from typing import Optional

from sacrebleu.metrics import BLEU, CHRF
from rouge_score import rouge_scorer


class UnicodeTokenizer:
    """Tokenize text while preserving Indic letters and combining marks."""

    def tokenize(self, text: str):
        text = unicodedata.normalize("NFKC", text or "").casefold()

        characters = []
        for char in text:
            category = unicodedata.category(char)

            if category[0] in ("L", "M", "N") or char.isspace():
                characters.append(char)
            else:
                characters.append(" ")

        return "".join(characters).split()


_bleu = BLEU()
_chrf = CHRF()
_rouge = rouge_scorer.RougeScorer(
    ["rougeL"],
    use_stemmer=False,
    tokenizer=UnicodeTokenizer(),
)


def normalize_text(text: Optional[str]) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.casefold().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    return text


def exact_match_score(prediction: str, reference: str) -> float:
    """Return 1.0 for normalized exact match, otherwise 0.0."""
    if not normalize_text(reference):
        return float("nan")

    return (
        1.0
        if normalize_text(prediction) == normalize_text(reference)
        else 0.0
    )


def translation_metrics(prediction: str, reference: str) -> dict:
    if not reference or not reference.strip():
        return {"bleu": None, "chrf": None}

    return {
        "bleu": round(
            float(_bleu.sentence_score(prediction, [reference]).score), 4
        ),
        "chrf": round(
            float(_chrf.sentence_score(prediction, [reference]).score), 4
        ),
    }


def summarization_metrics(prediction: str, reference: str) -> dict:
    if not reference or not reference.strip():
        return {"rougeL_f1": None}

    scores = _rouge.score(reference, prediction)

    return {
        "rougeL_f1": round(
            float(scores["rougeL"].fmeasure * 100), 4
        )
    }


def reasoning_metrics(prediction: str, reference: str) -> dict:
    if not reference or not reference.strip():
        return {"exact_match": None, "correct": None}

    match = re.match(
        r"^\s*([A-D])(?:[\).:\s]|$)",
        reference,
        flags=re.IGNORECASE,
    )

    if match:
        expected = match.group(1).upper()
        pred_match = re.search(
            r"(?<![A-Z])([A-D])(?:[\).:\s]|$)",
            prediction.strip(),
            flags=re.IGNORECASE,
        )
        correct = bool(
            pred_match and pred_match.group(1).upper() == expected
        )
        return {
            "exact_match": float(correct),
            "correct": correct,
        }

    score = exact_match_score(prediction, reference)

    return {
        "exact_match": None if score != score else score,
        "correct": None if score != score else bool(score),
    }


def basic_code_switch_checks(prediction: str) -> dict:
    """Only a smoke check; does not measure semantic quality."""
    answer = (prediction or "").strip()

    return {
        "non_empty": bool(answer),
        "word_count": len(answer.split()),
        "judge_score": None,
        "judge_verdict": None,
        "judge_reason": None,
    }