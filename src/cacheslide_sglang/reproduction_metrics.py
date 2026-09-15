"""Reference-answer metrics, deliberately separate from token consistency.

The paper names ROUGE-L *recall*, not its F score. Its tokenizer, stemming,
answer extraction and exact benchmark versions were not specified; these
explicit implementations are an auditable protocol, not a claimed match to
an unavailable paper evaluator. SWE-bench resolved rate needs sandbox tests.
"""

from __future__ import annotations

import re
import string
from collections import Counter

METRICS = frozenset(
    {"qa_f1", "exact_match", "rouge_l_recall", "hotpotqa_f1", "hotpotqa_em"}
)
HOTPOTQA_SCORER = {
    "source": "https://github.com/hotpotqa/hotpot/blob/"
    "fa3a36370899e1d85822de61e58c85ea19993154/hotpot_evaluate_v1.py",
    "source_sha256": "d35fc91a6db21d791dbdda11daf3856e9359f5701d54e3eefba20d88fecc02c0",
    "scope": "Independently implemented official answer F1/EM semantics only; "
    "not supporting-fact or joint scores and not a Reflexion success rate.",
}


def answer_tokens(text: str) -> list[str]:
    """SQuAD-style lowercase, ASCII punctuation/articles, whitespace tokens."""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\b(a|an|the)\b", " ", text).split()


def rouge_tokens(text: str) -> list[str]:
    """Explicit non-stemmed, lowercase Unicode word tokenizer."""
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def lcs_length(left: list[str], right: list[str]) -> int:
    """Exact longest common subsequence using O(min(n, m)) memory."""
    if len(left) < len(right):
        left, right = right, left
    previous = [0] * (len(right) + 1)
    for word in left:
        current = [0]
        for index, other in enumerate(right, start=1):
            current.append(
                previous[index - 1] + 1
                if word == other
                else max(previous[index], current[-1])
            )
        previous = current
    return previous[-1]


def score_answer(prediction: str, references: list[str], metric: str) -> float:
    """Best-reference score on [0, 1]; never a model-to-model agreement score."""
    if metric not in METRICS:
        raise ValueError("unsupported answer metric; SWE resolved needs real tests")
    if (
        not isinstance(prediction, str)
        or not isinstance(references, list)
        or not references
        or any(not isinstance(reference, str) for reference in references)
    ):
        raise ValueError("prediction and nonempty reference list must contain text")
    if metric.startswith("hotpotqa_") and len(references) != 1:
        raise ValueError("official HotpotQA answer metrics require one gold answer")
    scores = []
    for reference in references:
        tokenizer = rouge_tokens if metric == "rouge_l_recall" else answer_tokens
        predicted, gold = tokenizer(prediction), tokenizer(reference)
        if metric in {"exact_match", "hotpotqa_em"}:
            value = float(predicted == gold)
        elif metric == "hotpotqa_f1":
            # Hotpot does not grant partial credit for mismatched categorical
            # answers, and zero overlap (including two empty answers) scores 0.
            categorical = {"yes", "no", "noanswer"}
            mismatch = predicted != gold and (
                " ".join(predicted) in categorical or " ".join(gold) in categorical
            )
            overlap = sum((Counter(predicted) & Counter(gold)).values())
            value = (
                0.0
                if mismatch or not overlap
                else (2 * overlap / (len(predicted) + len(gold)))
            )
        elif not gold or not predicted:
            value = float(predicted == gold)
        elif metric == "rouge_l_recall":
            value = lcs_length(predicted, gold) / len(gold)
        else:
            overlap = sum((Counter(predicted) & Counter(gold)).values())
            value = 2 * overlap / (len(predicted) + len(gold))
        scores.append(value)
    return max(scores)
