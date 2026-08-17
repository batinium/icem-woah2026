"""Dataset loaders and normalization into the common I-CEM row schema."""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import replace
import json
import os
from pathlib import Path
import random
import re
from typing import Any
from urllib.request import urlopen

from .schema import Row


HATEXPLAIN_ID = "Hate-speech-CNERG/hatexplain"
HATECHECK_ID = "Paul/hatecheck"
CIVIL_COMMENTS_ID = "google/civil_comments"
TOXIC_SPANS_ID = "heegyu/toxic-spans"
HATEXPLAIN_DATA_URL = "https://raw.githubusercontent.com/punyajoy/HateXplain/master/Data/dataset.json"
HATEXPLAIN_SPLITS_URL = "https://raw.githubusercontent.com/punyajoy/HateXplain/master/Data/post_id_divisions.json"
CIVIL_COMMENTS_SCORE_FIELDS = (
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
    "sexual_explicit",
)
CIVIL_COMMENTS_TARGET_TERMS = {
    "asian": ("asian", "asians"),
    "black": ("black", "black people"),
    "christian": ("christian", "christians"),
    "disabled": ("disabled", "disability", "mental illness", "psychiatric"),
    "gay": ("gay", "lesbian", "homosexual", "lgbt", "queer"),
    "immigrant": ("immigrant", "immigrants", "refugee", "refugees"),
    "jewish": ("jew", "jews", "jewish"),
    "muslim": ("muslim", "muslims", "islam"),
    "trans": ("trans", "transgender"),
    "white": ("white", "white people"),
    "women": ("woman", "women", "female", "females"),
}


def load_rows_from_csv(
    path: Path,
    *,
    text_col: str,
    label_col: str | None = None,
    id_col: str | None = None,
) -> tuple[Row, ...]:
    """Load a generic CSV file into the common schema."""

    rows: list[Row] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if text_col not in fieldnames:
            raise ValueError(f"missing text column: {text_col}")
        if label_col and label_col not in fieldnames:
            raise ValueError(f"missing label column: {label_col}")
        if id_col and id_col not in fieldnames:
            raise ValueError(f"missing id column: {id_col}")
        for index, raw in enumerate(reader):
            row_id = raw.get(id_col or "", "") or str(index)
            label = raw.get(label_col or "", None) if label_col else None
            rows.append(
                Row(
                    row_id=row_id,
                    dataset=path.stem,
                    split="unknown",
                    text=raw[text_col],
                    label=label,
                    binary_label=label_to_binary(label),
                    metadata={"source_path": str(path)},
                )
            )
    return tuple(rows)


def load_builtin_smoke_rows() -> tuple[Row, ...]:
    """Small synthetic rows used when CSV mode is invoked without an input file."""

    examples = (
        ("smoke-0", "I do not agree with people saying Muslims should be banned.", "normal", 0),
        ("smoke-1", "Immigrants should be kicked out.", "hateful", 1),
        ("smoke-2", "She quoted Jews are vermin and condemned it.", "normal", 0),
        ("smoke-3", "Those people should leave now.", "hateful", 1),
    )
    return tuple(
        Row(
            row_id=row_id,
            dataset="csv_smoke",
            split="synthetic",
            text=text,
            label=label,
            binary_label=binary,
            target_groups=("muslim", "immigrant", "jew") if index < 3 else (),
            metadata={"synthetic_smoke": True},
        )
        for index, (row_id, text, label, binary) in enumerate(examples)
    )


def load_hatexplain_rows(*, split: str = "train") -> tuple[Row, ...]:
    try:
        return _load_hatexplain_rows_direct(split=split)
    except Exception as direct_exc:
        try:
            from datasets import load_dataset
        except ImportError as import_exc:  # pragma: no cover - depends on environment
            raise RuntimeError(f"failed to load HateXplain directly: {direct_exc}") from import_exc
        try:
            dataset = load_dataset(HATEXPLAIN_ID, split=split)
        except Exception as hf_exc:  # pragma: no cover - depends on environment/version
            raise RuntimeError(
                f"failed to load HateXplain directly ({direct_exc}) or through datasets ({hf_exc})"
            ) from hf_exc
        return _normalize_hatexplain_records(dataset, split=split)


def _load_hatexplain_rows_direct(*, split: str = "train") -> tuple[Row, ...]:
    cache_dir = Path(os.environ.get("HF_HOME", "data/hf_cache")) / "raw" / "hatexplain"
    dataset = _download_json(HATEXPLAIN_DATA_URL, cache_dir / "dataset.json")
    splits = _download_json(HATEXPLAIN_SPLITS_URL, cache_dir / "post_id_divisions.json")
    if split not in splits:
        raise RuntimeError(f"HateXplain split not found: {split}")
    records = []
    for tweet_id in splits[split]:
        info = dataset[str(tweet_id)]
        records.append(
            {
                "id": str(tweet_id),
                "annotators": info.get("annotators"),
                "rationales": info.get("rationales"),
                "post_tokens": info.get("post_tokens"),
            }
        )
    return _normalize_hatexplain_records(records, split=split)


def _normalize_hatexplain_records(dataset: Iterable[Any], *, split: str) -> tuple[Row, ...]:
    rows: list[Row] = []
    for index, raw in enumerate(dataset):
        tokens = raw.get("post_tokens") or raw.get("tokens") or raw.get("text")
        if isinstance(tokens, str):
            text = tokens
            token_list: list[str] = tokens.split()
        else:
            token_list = [str(token) for token in (tokens or [])]
            text = reconstruct_text(token_list)
        annotators = raw.get("annotators") or {}
        labels = _field_values(annotators, "label") or _field_values(raw, "label")
        label = majority_vote(labels)
        targets = tuple(sorted(set(_target_values(annotators) or _target_values(raw))))
        rationales = raw.get("rationales") or raw.get("rationale")
        rationale_mask = combine_rationales(rationales, token_count=len(token_list))
        rows.append(
            Row(
                row_id=str(raw.get("post_id") or raw.get("id") or index),
                dataset="hatexplain",
                split=split,
                text=text,
                label=label,
                binary_label=label_to_binary(label),
                target_groups=targets,
                rationale_token_mask=rationale_mask,
                metadata={"hf_dataset": HATEXPLAIN_ID},
            )
        )
    return tuple(rows)


def _download_json(url: str, path: Path) -> Any:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urlopen(url, timeout=60) as response:
            payload = response.read()
        path.write_bytes(payload)
    return json.loads(path.read_text(encoding="utf-8"))


def load_hatecheck_rows(*, split: str | None = None) -> tuple[Row, ...]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("datasets is required to load HateCheck") from exc

    loaded = load_dataset(HATECHECK_ID)
    split_name = split or ("test" if "test" in loaded else next(iter(loaded.keys())))
    dataset = loaded[split_name]
    rows: list[Row] = []
    for index, raw in enumerate(dataset):
        text = str(raw.get("test_case") or raw.get("text") or raw.get("case") or "")
        label = raw.get("label_gold") or raw.get("label") or raw.get("gold_label")
        target = raw.get("target_ident") or raw.get("target_group") or raw.get("target")
        functionality = raw.get("functionality") or raw.get("functional_test") or raw.get("case_type")
        rows.append(
            Row(
                row_id=str(raw.get("id") or index),
                dataset="hatecheck",
                split=split_name,
                text=text,
                label=label,
                binary_label=label_to_binary(label),
                target_groups=tuple(_flatten_values(target)),
                functionality=str(functionality) if functionality is not None else None,
                metadata={"hf_dataset": HATECHECK_ID},
            )
        )
    return tuple(rows)


def load_civil_comments_rows(*, split: str = "validation", toxicity_threshold: float = 0.5) -> tuple[Row, ...]:
    """Load Civil Comments for out-of-domain toxicity transfer checks.

    The Hugging Face `google/civil_comments` mirror exposes toxicity score
    columns but not the extended identity-mention columns. We therefore use
    toxicity as the gold utility label and lexical identity terms only as
    non-gold context cues.
    """

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("datasets is required to load Civil Comments") from exc

    dataset = load_dataset(CIVIL_COMMENTS_ID, split=split)
    rows: list[Row] = []
    for index, raw in enumerate(dataset):
        text = str(raw.get("text") or "")
        toxicity = _float_or_zero(raw.get("toxicity"))
        binary = int(toxicity >= toxicity_threshold)
        label = "toxic" if binary else "non-toxic"
        score_metadata = {field: _float_or_zero(raw.get(field)) for field in CIVIL_COMMENTS_SCORE_FIELDS}
        rows.append(
            Row(
                row_id=str(raw.get("id") or index),
                dataset="civil_comments",
                split=split,
                text=text,
                label=label,
                binary_label=binary,
                target_groups=_civil_comments_targets(text),
                metadata={
                    "hf_dataset": CIVIL_COMMENTS_ID,
                    "toxicity_threshold": toxicity_threshold,
                    "target_groups_source": "lexical_identity_terms_not_gold",
                    **score_metadata,
                },
            )
        )
    return tuple(rows)


def sample_rows(rows: Sequence[Row], sample_size: int | None, *, seed: int) -> tuple[Row, ...]:
    if sample_size is None or sample_size >= len(rows):
        return tuple(rows)
    rng = random.Random(seed)
    indexes = sorted(rng.sample(range(len(rows)), sample_size))
    return tuple(rows[index] for index in indexes)


def dataset_summary(rows: Sequence[Row]) -> dict[str, Any]:
    labels = Counter(str(row.label) for row in rows if row.label is not None)
    binary = Counter(str(row.binary_label) for row in rows if row.binary_label is not None)
    datasets = Counter(row.dataset for row in rows)
    splits = Counter(row.split for row in rows)
    targets = Counter(target for row in rows for target in row.target_groups)
    return {
        "row_count": len(rows),
        "datasets": dict(sorted(datasets.items())),
        "splits": dict(sorted(splits.items())),
        "labels": dict(labels.most_common()),
        "binary_labels": dict(sorted(binary.items())),
        "target_groups_top20": dict(targets.most_common(20)),
        "rows_with_rationales": sum(1 for row in rows if row.rationale_token_mask),
    }


def _parse_position_field(value: Any) -> list[int]:
    """Toxic Spans stores `position` as a list or a stringified list of char offsets."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    text = str(value).strip()
    if not text or text in {"[]", "nan"}:
        return []
    import ast

    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return [int(part) for part in re.findall(r"\d+", text)]
    if isinstance(parsed, (list, tuple)):
        return [int(item) for item in parsed]
    return [int(parsed)]


def load_toxic_spans_rows(*, split: str = "train") -> tuple[Row, ...]:
    """Load SemEval-2021 Task 5 Toxic Spans Detection (CC0-1.0).

    Each row carries human-annotated character offsets of the toxic span in
    ``position``. We convert those gold spans into a token-level rationale mask
    aligned to ``tokenize_with_offsets`` so they can validate the lexical
    context proxies with an external human signal, independent of HateXplain.
    """

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("datasets is required to load Toxic Spans") from exc

    from .tokenizer import tokenize_with_offsets

    loaded = load_dataset(TOXIC_SPANS_ID)
    split_name = split if split in loaded else next(iter(loaded.keys()))
    dataset = loaded[split_name]
    rows: list[Row] = []
    for index, raw in enumerate(dataset):
        text = str(raw.get("text_of_post") or raw.get("text") or "")
        if not text.strip():
            continue
        toxic_chars = {int(pos) for pos in _parse_position_field(raw.get("position"))}
        tokens = tokenize_with_offsets(text)
        if toxic_chars:
            mask: tuple[int, ...] | None = tuple(
                1 if any(char in toxic_chars for char in range(token.start, token.end)) else 0
                for token in tokens
            )
            if not any(mask):
                mask = None
        else:
            mask = None
        label = raw.get("toxic")
        rows.append(
            Row(
                row_id=str(raw.get("id") or index),
                dataset="toxic_spans",
                split=split_name,
                text=text,
                label=label,
                binary_label=label_to_binary(label),
                target_groups=(),
                rationale_token_mask=mask,
                metadata={"hf_dataset": TOXIC_SPANS_ID},
            )
        )
    return tuple(rows)


def with_replaced_text(row: Row, text: str, metadata: dict[str, Any]) -> Row:
    return replace(row, text=text, metadata={**dict(row.metadata), **metadata})


def reconstruct_text(tokens: Sequence[str]) -> str:
    text = " ".join(tokens)
    text = re.sub(r"\s+([,.;:!?%])", r"\1", text)
    text = re.sub(r"([(])\s+", r"\1", text)
    text = re.sub(r"\s+([)])", r"\1", text)
    text = text.replace("`` ", '"').replace(" ''", '"')
    return text.strip()


def majority_vote(values: Iterable[Any]) -> str | int | None:
    normalized = [value for value in values if value is not None and str(value).lower() not in {"", "none", "nan"}]
    if not normalized:
        return None
    return Counter(normalized).most_common(1)[0][0]


def label_to_binary(label: Any) -> int | None:
    if label is None:
        return None
    if isinstance(label, bool):
        return int(label)
    if isinstance(label, (int, float)) and not isinstance(label, bool):
        if int(label) in {0, 1}:
            return int(label)
    lowered = str(label).strip().lower().replace("_", " ")
    if lowered in {"normal", "non hate", "non-hate", "non hateful", "not hateful", "none", "clean", "0"}:
        return 0
    if any(term in lowered for term in ("hate", "hateful", "offensive", "toxic", "abusive", "threat")):
        if lowered.startswith("non") or "not hateful" in lowered:
            return 0
        return 1
    if lowered in {"1", "true", "yes"}:
        return 1
    return None


def combine_rationales(rationales: Any, *, token_count: int) -> tuple[int, ...] | None:
    if not rationales or token_count <= 0:
        return None
    if isinstance(rationales, list) and rationales and all(isinstance(value, int) for value in rationales):
        values = [1 if int(value) else 0 for value in rationales[:token_count]]
        return tuple(values + [0] * max(0, token_count - len(values)))
    masks = [mask for mask in rationales if isinstance(mask, list)] if isinstance(rationales, list) else []
    if not masks:
        return None
    combined: list[int] = []
    for index in range(token_count):
        votes = [int(mask[index]) for mask in masks if index < len(mask)]
        combined.append(1 if votes and sum(votes) / len(votes) >= 0.5 else 0)
    return tuple(combined)


def _field_values(container: Any, field: str) -> list[Any]:
    if isinstance(container, dict):
        value = container.get(field)
    elif isinstance(container, list):
        output: list[Any] = []
        for item in container:
            output.extend(_field_values(item, field))
        return output
    else:
        value = getattr(container, field, None)
    return list(_flatten_values(value))


def _target_values(container: Any) -> list[str]:
    values: list[str] = []
    for field in ("target", "targets", "target_groups", "target_group"):
        values.extend(str(value).lower() for value in _field_values(container, field))
    return [value for value in values if value and value not in {"none", "nan", "other"}]


def _flatten_values(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, int, float, bool)):
        return [value]
    if isinstance(value, dict):
        output: list[Any] = []
        for nested in value.values():
            output.extend(_flatten_values(nested))
        return output
    if isinstance(value, Iterable):
        output = []
        for nested in value:
            output.extend(_flatten_values(nested))
        return output
    return [value]


def _float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _civil_comments_targets(text: str) -> tuple[str, ...]:
    lowered = text.lower()
    targets: list[str] = []
    for label, terms in CIVIL_COMMENTS_TARGET_TERMS.items():
        if any(re.search(rf"(?<!\w){re.escape(term)}(?!\w)", lowered) for term in terms):
            targets.append(label)
    return tuple(targets)
