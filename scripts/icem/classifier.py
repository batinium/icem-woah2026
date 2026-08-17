"""Classifier protocol and simple deterministic fallback for experiments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from .schema import Row


class HsdClassifier(Protocol):
    def predict_proba(self, texts: Sequence[str]) -> list[float]:
        """Return harmful-speech scores in [0, 1]."""


class CachedHsdClassifier:
    """Memoize classifier scores for repeated texts within one experiment run."""

    def __init__(self, classifier: HsdClassifier) -> None:
        self.classifier = classifier
        self.cache: dict[str, float] = {}

    def predict_proba(self, texts: Sequence[str]) -> list[float]:
        ordered_texts = list(texts)
        missing = list(dict.fromkeys(text for text in ordered_texts if text not in self.cache))
        if missing:
            scores = self.classifier.predict_proba(missing)
            self.cache.update(zip(missing, scores, strict=True))
        return [self.cache[text] for text in ordered_texts]


class LexiconHsdClassifier:
    """Tiny fallback classifier for smoke tests, not a paper-quality model."""

    def __init__(self, cues: Sequence[str] | None = None) -> None:
        self.cues = tuple(
            cue.lower()
            for cue in (
                cues
                or (
                    "ban",
                    "banned",
                    "deport",
                    "leave",
                    "vermin",
                    "hate",
                    "kill",
                    "kicked out",
                    "inferior",
                    "terrorist",
                    "should leave",
                    "should be banned",
                )
            )
        )

    def predict_proba(self, texts: Sequence[str]) -> list[float]:
        scores: list[float] = []
        for text in texts:
            lowered = text.lower()
            hits = sum(1 for cue in self.cues if cue in lowered)
            scores.append(min(1.0, 0.05 + hits * 0.28))
        return scores


class SklearnTfidfHsdClassifier:
    """Fast frozen baseline used when scikit-learn is available."""

    def __init__(self, rows: Sequence[Row], *, seed: int = 17) -> None:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError("scikit-learn is required for the tfidf classifier") from exc

        train_rows = [row for row in rows if row.binary_label in {0, 1}]
        labels = [int(row.binary_label or 0) for row in train_rows]
        if len(set(labels)) < 2:
            raise RuntimeError("tfidf classifier requires at least one positive and one negative label")
        self.pipeline = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=20000),
            LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced"),
        )
        self.pipeline.fit([row.text for row in train_rows], labels)

    def predict_proba(self, texts: Sequence[str]) -> list[float]:
        probabilities = self.pipeline.predict_proba(list(texts))
        classes = list(self.pipeline.classes_)
        positive_index = classes.index(1)
        return [float(row[positive_index]) for row in probabilities]


class TransformersHsdClassifier:
    """Hugging Face sequence-classification wrapper.

    The wrapper is optional so the research scripts can still smoke-test in
    minimal environments.
    """

    def __init__(self, model_id: str, *, batch_size: int = 16, device: str = "auto") -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError("transformers and torch are required for Hugging Face classifiers") from exc

        self.torch = torch
        self.batch_size = batch_size
        self.device = _resolve_torch_device(torch, device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()
        self.id2label = _normalized_id2label(getattr(self.model.config, "id2label", None))
        self.positive_index = _positive_label_index(self.id2label)
        self.output_activation = _output_activation(self.id2label, getattr(self.model.config, "problem_type", None))

    def predict_proba(self, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        scores: list[float] = []
        with self.torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = list(texts[start : start + self.batch_size])
                encoded = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                logits = self.model(**encoded).logits
                if logits.shape[-1] == 1:
                    batch_scores = self.torch.sigmoid(logits[:, 0])
                elif self.output_activation == "sigmoid":
                    batch_scores = self.torch.sigmoid(logits[:, self.positive_index])
                else:
                    batch_scores = self.torch.softmax(logits, dim=-1)[:, self.positive_index]
                scores.extend(float(value) for value in batch_scores.detach().cpu().tolist())
        return scores


@dataclass(frozen=True)
class ClassifierBuildResult:
    classifier: HsdClassifier
    name: str
    details: str


def _normalized_id2label(id2label: object) -> dict[int, str]:
    if not isinstance(id2label, dict):
        return {}
    normalized: dict[int, str] = {}
    for key, value in id2label.items():
        try:
            normalized[int(key)] = str(value)
        except (TypeError, ValueError):
            continue
    return dict(sorted(normalized.items()))


def _positive_label_index(id2label: object) -> int:
    labels = _normalized_id2label(id2label)
    if labels:
        lowered_labels = {index: label.lower() for index, label in labels.items()}
        for index, label in labels.items():
            lowered = lowered_labels[index]
            if any(term in lowered for term in ("hate", "toxic", "offensive", "abusive", "harm")) and not any(
                neg in lowered for neg in ("non", "not", "normal", "clean")
            ):
                return index
        if len(labels) == 2:
            return max(labels)
    return 1


def _resolve_torch_device(torch: object, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _output_activation(id2label: object, problem_type: object) -> str:
    if problem_type == "multi_label_classification":
        return "sigmoid"
    labels = {label.lower() for label in _normalized_id2label(id2label).values()}
    toxicity_labels = {
        "toxicity",
        "severe_toxicity",
        "obscene",
        "identity_attack",
        "insult",
        "threat",
        "sexual_explicit",
    }
    if len(labels & toxicity_labels) >= 4:
        return "sigmoid"
    return "softmax"


def _cached(classifier: HsdClassifier) -> CachedHsdClassifier:
    return CachedHsdClassifier(classifier)


def build_classifier(
    name: str,
    rows: Sequence[Row],
    *,
    seed: int = 17,
    batch_size: int = 16,
    device: str = "auto",
) -> ClassifierBuildResult:
    """Build a frozen classifier, falling back to lexicon when requested/needed."""

    normalized = name.strip() if name else "tfidf"
    if normalized == "lexicon":
        return ClassifierBuildResult(_cached(LexiconHsdClassifier()), "lexicon", "deterministic cue lexicon")
    if normalized == "tfidf":
        try:
            return ClassifierBuildResult(
                _cached(SklearnTfidfHsdClassifier(rows, seed=seed)),
                "tfidf",
                "TF-IDF logistic regression trained once on loaded rows; in-memory prediction cache enabled",
            )
        except RuntimeError as exc:
            return ClassifierBuildResult(
                _cached(LexiconHsdClassifier()),
                "lexicon",
                f"tfidf unavailable; fell back to deterministic cue lexicon ({exc}); in-memory prediction cache enabled",
            )
    classifier = TransformersHsdClassifier(normalized, batch_size=batch_size, device=device)
    positive_label = classifier.id2label.get(classifier.positive_index, "unknown")
    return ClassifierBuildResult(
        _cached(classifier),
        normalized,
        (
            f"Hugging Face sequence classifier {normalized}; "
            f"harmful_positive_index={classifier.positive_index}; "
            f"harmful_positive_label={positive_label}; "
            f"id2label={classifier.id2label}; "
            f"device={classifier.device}; "
            f"batch_size={classifier.batch_size}; "
            f"output_activation={classifier.output_activation}; "
            "in-memory prediction cache enabled"
        ),
    )
