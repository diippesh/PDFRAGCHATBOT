"""RAGAS evaluation for recorded question, answer, and retrieved-context samples."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path

from src.config import Settings
from src.exceptions import EvaluationError
from src.logging_config import get_logger

logger = get_logger(__name__)


def _normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", question.lower())).strip()


def _load_ground_truth() -> dict[str, dict[str, object]]:
    path = Path(__file__).resolve().parent.parent / "data" / "ground_truth.json"
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"Could not load ground truth file '{path}': {exc}") from exc
    return {_normalize_question(record["user_input"]): record for record in records}


def _attach_references(
    samples: Sequence[dict[str, object]], indexed_sources: Sequence[str]
) -> tuple[list[dict], list[dict]]:
    ground_truth = _load_ground_truth()
    indexed_names = {Path(source).name.casefold() for source in indexed_sources}
    matched, unmatched = [], []
    for sample in samples:
        record = ground_truth.get(_normalize_question(str(sample["user_input"])))
        if record and str(record["source"]).casefold() in indexed_names:
            matched.append(
                {
                    **sample,
                    "reference": record["reference"],
                    "reference_contexts": record["reference_contexts"],
                }
            )
        else:
            unmatched.append(dict(sample))
    return matched, unmatched


def evaluate_samples(
    samples: Sequence[dict[str, object]],
    settings: Settings,
    indexed_sources: Sequence[str] = (),
) -> dict[str, float]:
    """Evaluate samples with reference-based metrics when a ground truth exists."""
    if not samples:
        raise EvaluationError("There are no completed question-and-answer pairs to evaluate.")

    try:
        from langchain_groq import ChatGroq
        from ragas import evaluate
        from ragas.dataset_schema import EvaluationDataset
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            AnswerCorrectness,
            AnswerRelevancy,
            ContextRecall,
            Faithfulness,
            LLMContextPrecisionWithoutReference,
            LLMContextPrecisionWithReference,
        )
    except ImportError as exc:
        raise EvaluationError(
            "RAGAS dependencies are not installed. Run 'pip install -r requirements.txt'."
        ) from exc

    if not settings.groq_api_key:
        raise EvaluationError("GROQ_API_KEY is required to run RAGAS evaluation.")

    try:
        llm = ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            temperature=0,
            max_tokens=settings.llm_max_tokens,
        )
        matched, unmatched = _attach_references(samples, indexed_sources)
        scores: dict[str, float] = {
            "ground_truth_matches": float(len(matched)),
            "reference_free_samples": float(len(unmatched)),
        }
        embeddings = LangchainEmbeddingsWrapper(
            _load_embeddings(settings.embedding_model, settings.embedding_device)
        )
        wrapped_llm = LangchainLLMWrapper(llm)

        if matched:
            result = evaluate(
                EvaluationDataset.from_list(matched),
                metrics=[
                    Faithfulness(),
                    AnswerRelevancy(),
                    AnswerCorrectness(),
                    ContextRecall(),
                    LLMContextPrecisionWithReference(),
                ],
                llm=wrapped_llm,
                embeddings=embeddings,
            )
            scores.update(
                {
                    f"ground_truth_{name}": float(value)
                    for name, value in result.to_pandas().mean(numeric_only=True).items()
                }
            )

        if unmatched:
            result = evaluate(
                EvaluationDataset.from_list(unmatched),
                metrics=[
                    Faithfulness(),
                    AnswerRelevancy(),
                    LLMContextPrecisionWithoutReference(),
                ],
                llm=wrapped_llm,
                embeddings=embeddings,
            )
            scores.update(
                {
                    f"reference_free_{name}": float(value)
                    for name, value in result.to_pandas().mean(numeric_only=True).items()
                }
            )
        return scores
    except Exception as exc:
        logger.exception("RAGAS evaluation failed")
        raise EvaluationError(f"RAGAS evaluation failed: {exc}") from exc


def _load_embeddings(model_name: str, device: str):
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
