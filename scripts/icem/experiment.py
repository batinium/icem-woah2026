"""Experiment entrypoint for I-CEM pilot runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .classifier import build_classifier
from .context_rules import cue_token_indexes
from .datasets import (
    dataset_summary,
    load_builtin_smoke_rows,
    load_civil_comments_rows,
    load_toxic_spans_rows,
    load_hatecheck_rows,
    load_hatexplain_rows,
    load_rows_from_csv,
    sample_rows,
    with_replaced_text,
)
from .importance import occlusion_importance, select_anchor_indexes
from .metrics import aggregate_variant_metrics
from .release_policy import DEFAULT_VARIANTS, release_variant
from .report import main_result_table, qualitative_examples_table, write_metrics_csv
from .schema import ImportanceScore, ReleaseResult, Row, Span
from .spans import detect_identifier_spans
from .synthetic_pii import inject_synthetic_pii
from .tokenizer import tokenize_with_offsets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run I-CEM research experiments.")
    parser.add_argument("--dataset", choices=["hatexplain", "hatecheck", "civil_comments", "toxic_spans", "csv"], default="csv")
    parser.add_argument("--input-csv", type=Path)
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col")
    parser.add_argument("--id-col")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--classifier", default="tfidf")
    parser.add_argument("--classifier-batch-size", type=int, default=16)
    parser.add_argument("--classifier-device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--importance-top-k", type=int, default=5)
    parser.add_argument("--importance-min-delta", type=float, default=0.02)
    parser.add_argument("--max-anchor-fraction", type=float, default=0.30)
    parser.add_argument("--window-radius", type=int, default=2)
    parser.add_argument("--target-harm-max-gap", type=int, default=8)
    parser.add_argument("--stance-harm-max-gap", type=int, default=8)
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--include-hatecheck", action="store_true")
    parser.add_argument("--inject-synthetic-pii", dest="inject_synthetic_pii", action="store_true", default=True)
    parser.add_argument("--no-inject-synthetic-pii", dest="inject_synthetic_pii", action="store_false")
    parser.add_argument("--full-output-dir", type=Path, default=Path("data/outputs"))
    parser.add_argument(
        "--manual-review-sample-size",
        type=int,
        default=0,
        help="Number of transformed-text review rows to write; 0 disables this legacy raw review artifact.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.output_dir.exists() and not args.overwrite:
        parser.error(f"output directory already exists: {args.output_dir}")

    variants = _parse_variants(args.variants, parser)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(args, parser)
    rows = sample_rows(rows, args.sample_size, seed=args.seed)
    if args.include_hatecheck and args.dataset != "hatecheck":
        hatecheck_rows = sample_rows(load_hatecheck_rows(), min(args.sample_size or 50, 100), seed=args.seed)
        rows = tuple(rows) + tuple(hatecheck_rows)
    if args.inject_synthetic_pii:
        rows = _inject_rows(rows, seed=args.seed)

    classifier_result = build_classifier(
        args.classifier,
        rows,
        seed=args.seed,
        batch_size=args.classifier_batch_size,
        device=args.classifier_device,
    )
    classifier = classifier_result.classifier

    results_by_variant: dict[str, list[ReleaseResult]] = {variant: [] for variant in variants}
    all_importance: dict[str, tuple[ImportanceScore, ...]] = {}
    identifier_spans_by_row: dict[str, tuple[Span, ...]] = {}

    for row_index, row in enumerate(rows):
        tokens = tokenize_with_offsets(row.text)
        gold_spans = _metadata_spans(row)
        identifier_spans = detect_identifier_spans(row.text, gold_spans=gold_spans)
        identifier_spans_by_row[row.row_id] = identifier_spans
        importance_scores = occlusion_importance(
            row.text,
            tokens,
            classifier,
            excluded_spans=identifier_spans,
        )
        all_importance[row.row_id] = importance_scores
        anchors = select_anchor_indexes(
            importance_scores,
            top_k=args.importance_top_k,
            min_delta=args.importance_min_delta,
            max_anchor_fraction=args.max_anchor_fraction,
            token_count=len(tokens),
        )
        if not anchors:
            anchors = _fallback_anchor_indexes(tokens, row.target_groups, top_k=args.importance_top_k)
        for variant in variants:
            result = release_variant(
                variant=variant,
                row_id=row.row_id,
                text=row.text,
                tokens=tokens,
                anchor_indexes=anchors,
                identifier_spans=identifier_spans,
                window_radius=args.window_radius,
                target_groups=row.target_groups,
                target_harm_max_gap=args.target_harm_max_gap,
                stance_harm_max_gap=args.stance_harm_max_gap,
                importance_scores=importance_scores,
            )
            results_by_variant[variant].append(result)
        if row_index and row_index % 100 == 0:
            print(f"processed {row_index}/{len(rows)} rows", flush=True)

    raw_scores = classifier.predict_proba([result.released_text for result in results_by_variant["raw"]])
    metrics_by_variant: dict[str, dict[str, Any]] = {}
    released_scores_by_variant: dict[str, list[float]] = {}
    for variant, results in results_by_variant.items():
        released_scores = classifier.predict_proba([result.released_text for result in results])
        released_scores_by_variant[variant] = released_scores
        metrics_by_variant[variant] = aggregate_variant_metrics(
            rows,
            results,
            released_scores=released_scores,
            raw_scores=raw_scores,
        )

    config = _serializable_config(args, variants, classifier_result.name, classifier_result.details)
    _write_json(args.output_dir / "config.json", config)
    _write_json(args.output_dir / "dataset_summary.json", dataset_summary(rows))
    _write_json(args.output_dir / "variant_metrics.json", metrics_by_variant)
    write_metrics_csv(args.output_dir / "variant_metrics.csv", metrics_by_variant)
    _write_privacy_breakdown(args.output_dir / "privacy_breakdown.csv", metrics_by_variant)
    _write_context_breakdown(
        args.output_dir / "context_breakdown.csv",
        rows,
        results_by_variant,
        released_scores_by_variant,
        raw_scores,
    )
    if args.manual_review_sample_size > 0:
        _write_manual_review_sample(
            args.output_dir / "manual_review_sample.csv",
            rows,
            results_by_variant,
            sample_size=args.manual_review_sample_size,
            seed=args.seed,
        )
    (args.output_dir / "main_table.md").write_text(main_result_table(metrics_by_variant), encoding="utf-8")
    (args.output_dir / "qualitative_examples.md").write_text(qualitative_examples_table(), encoding="utf-8")
    _write_full_outputs(args.full_output_dir, args.output_dir.name, rows, results_by_variant, identifier_spans_by_row)
    _write_json(
        args.output_dir / "run_manifest.json",
        {
            "aggregate_output_dir": str(args.output_dir),
            "full_output_dir": str(args.full_output_dir / args.output_dir.name),
            "row_count": len(rows),
            "variants": variants,
            "classifier": classifier_result.name,
        },
    )
    _write_result_readme(args.output_dir, config, metrics_by_variant)
    return 0


def _load_rows(args: argparse.Namespace, parser: argparse.ArgumentParser) -> tuple[Row, ...]:
    try:
        if args.dataset == "csv":
            if args.input_csv is None:
                return load_builtin_smoke_rows()
            return load_rows_from_csv(
                args.input_csv,
                text_col=args.text_col,
                label_col=args.label_col,
                id_col=args.id_col,
            )
        if args.dataset == "hatexplain":
            return load_hatexplain_rows()
        if args.dataset == "hatecheck":
            return load_hatecheck_rows()
        if args.dataset == "civil_comments":
            return load_civil_comments_rows()
        if args.dataset == "toxic_spans":
            return load_toxic_spans_rows()
    except Exception as exc:
        parser.error(str(exc))
    parser.error(f"unsupported dataset: {args.dataset}")
    raise AssertionError("unreachable")


def _parse_variants(value: str, parser: argparse.ArgumentParser) -> tuple[str, ...]:
    variants = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = [variant for variant in variants if variant not in DEFAULT_VARIANTS]
    if unknown:
        parser.error(f"unknown variant(s): {', '.join(unknown)}")
    if "raw" not in variants:
        parser.error("variants must include raw so flip rates can be computed")
    return variants


def _inject_rows(rows: tuple[Row, ...], *, seed: int) -> tuple[Row, ...]:
    injected_rows: list[Row] = []
    for index, row in enumerate(rows):
        injected = inject_synthetic_pii(row.text, seed=seed, row_index=index)
        spans = [_span_to_dict(span) for span in injected.spans]
        injected_rows.append(
            with_replaced_text(
                row,
                injected.text,
                {
                    "synthetic_pii_template": injected.template,
                    "synthetic_pii_spans": spans,
                },
            )
        )
    return tuple(injected_rows)


def _fallback_anchor_indexes(tokens: tuple[Any, ...], target_groups: tuple[str, ...], *, top_k: int) -> tuple[int, ...]:
    harm = cue_token_indexes(tokens, {"HARM_CUE"}, target_groups)
    target = cue_token_indexes(tokens, {"TARGET_CUE"}, target_groups)
    anchors = tuple(dict.fromkeys((*harm, *target)))
    return anchors[:top_k]


def _metadata_spans(row: Row) -> tuple[Span, ...]:
    spans: list[Span] = []
    for raw in row.metadata.get("synthetic_pii_spans", []):
        spans.append(_span_from_dict(raw))
    return tuple(spans)


def _span_from_dict(raw: dict[str, Any]) -> Span:
    return Span(
        start=int(raw["start"]),
        end=int(raw["end"]),
        label=str(raw["label"]),
        source=str(raw.get("source", "synthetic")),
        score=float(raw.get("score", 1.0)),
        replacement=raw.get("replacement"),
    )


def _span_to_dict(span: Span) -> dict[str, Any]:
    return {
        "start": span.start,
        "end": span.end,
        "label": span.label,
        "source": span.source,
        "score": span.score,
        "replacement": span.replacement,
    }


def _importance_to_dict(score: ImportanceScore) -> dict[str, Any]:
    return {
        "token_index": score.token_index,
        "token_text": score.token_text,
        "delta": score.delta,
        "baseline_score": score.baseline_score,
        "perturbed_score": score.perturbed_score,
    }


def _release_to_dict(result: ReleaseResult) -> dict[str, Any]:
    return {
        "row_id": result.row_id,
        "variant": result.variant,
        "source_text": result.source_text,
        "released_text": result.released_text,
        "kept_spans": [_span_to_dict(span) for span in result.kept_spans],
        "masked_spans": [_span_to_dict(span) for span in result.masked_spans],
        "warnings": list(result.warnings),
        "metadata": dict(result.metadata),
    }


def _row_to_dict(row: Row) -> dict[str, Any]:
    return {
        "row_id": row.row_id,
        "dataset": row.dataset,
        "split": row.split,
        "text": row.text,
        "label": row.label,
        "binary_label": row.binary_label,
        "target_groups": list(row.target_groups),
        "functionality": row.functionality,
        "metadata": dict(row.metadata),
    }


def _write_full_outputs(
    base_dir: Path,
    run_name: str,
    rows: tuple[Row, ...],
    results_by_variant: dict[str, list[ReleaseResult]],
    identifier_spans_by_row: dict[str, tuple[Span, ...]],
) -> None:
    output_dir = base_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "release_rows.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            variants = {
                variant: _release_to_dict(next(result for result in results if result.row_id == row.row_id))
                for variant, results in results_by_variant.items()
            }
            payload = {
                "row": _row_to_dict(row),
                "identifier_spans": [_span_to_dict(span) for span in identifier_spans_by_row.get(row.row_id, ())],
                "variants": variants,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_privacy_breakdown(path: Path, metrics_by_variant: dict[str, dict[str, Any]]) -> None:
    fields = [
        "variant",
        "row_count",
        "direct_pii_gold_total",
        "direct_pii_residual_count",
        "direct_pii_residual_rate",
        "direct_pii_removed_recall",
        "quasi_identifier_gold_total",
        "quasi_identifier_residual_count",
        "quasi_identifier_residual_rate",
        "detected_direct_identifier_residual_count",
        "detected_quasi_identifier_residual_count",
        "detected_source_marker_residual_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for variant, metrics in metrics_by_variant.items():
            writer.writerow({"variant": variant, **{field: metrics.get(field) for field in fields if field != "variant"}})


def _write_context_breakdown(
    path: Path,
    rows: tuple[Row, ...],
    results_by_variant: dict[str, list[ReleaseResult]],
    released_scores_by_variant: dict[str, list[float]],
    raw_scores: list[float],
) -> None:
    fields = [
        "group_type",
        "group",
        "variant",
        "row_count",
        "positive_f1",
        "prediction_flip_rate_raw",
        "retained_token_pct",
        "retained_char_pct",
        "target_cue_preservation",
        "harm_cue_preservation",
        "negation_cue_preservation",
        "quotation_cue_preservation",
        "counterspeech_cue_preservation",
        "target_harm_pair_preservation",
        "stance_harm_relation_preservation",
        "rationale_overlap",
    ]
    groups = _metric_groups(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for group_type, group, indexes in groups:
            grouped_rows = tuple(rows[index] for index in indexes)
            grouped_raw_scores = [raw_scores[index] for index in indexes]
            for variant, results in results_by_variant.items():
                grouped_results = [results[index] for index in indexes]
                grouped_scores = [released_scores_by_variant[variant][index] for index in indexes]
                metrics = aggregate_variant_metrics(
                    grouped_rows,
                    grouped_results,
                    released_scores=grouped_scores,
                    raw_scores=grouped_raw_scores,
                )
                writer.writerow(
                    {
                        "group_type": group_type,
                        "group": group,
                        "variant": variant,
                        **{field: metrics.get(field) for field in fields if field not in {"group_type", "group", "variant"}},
                    }
                )


def _metric_groups(rows: tuple[Row, ...]) -> list[tuple[str, str, tuple[int, ...]]]:
    grouped: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault(("dataset", row.dataset), []).append(index)
        if row.functionality:
            grouped.setdefault(("functionality", row.functionality), []).append(index)
    return [
        (group_type, group, tuple(indexes))
        for (group_type, group), indexes in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1]))
    ]


def _write_manual_review_sample(
    path: Path,
    rows: tuple[Row, ...],
    results_by_variant: dict[str, list[ReleaseResult]],
    *,
    sample_size: int,
    seed: int,
) -> None:
    fields = [
        "review_id",
        "row_id",
        "dataset",
        "split",
        "label",
        "binary_label",
        "functionality",
        "target_groups",
        "variant",
        "released_text",
        "label_preserved",
        "target_preserved",
        "harmful_cue_preserved",
        "stance_preserved",
        "context_sufficient_for_review",
        "misleading_after_reduction",
        "obvious_privacy_risk_remaining",
        "review_notes",
    ]
    rng = __import__("random").Random(seed)
    indexes = list(range(len(rows)))
    if sample_size > 0 and sample_size < len(indexes):
        indexes = sorted(rng.sample(indexes, sample_size))
    variants = [variant for variant in results_by_variant if variant != "raw"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        review_id = 0
        for index in indexes:
            row = rows[index]
            for variant in variants:
                result = results_by_variant[variant][index]
                review_id += 1
                writer.writerow(
                    {
                        "review_id": review_id,
                        "row_id": row.row_id,
                        "dataset": row.dataset,
                        "split": row.split,
                        "label": row.label,
                        "binary_label": row.binary_label,
                        "functionality": row.functionality,
                        "target_groups": ";".join(row.target_groups),
                        "variant": variant,
                        "released_text": result.released_text,
                        "label_preserved": "",
                        "target_preserved": "",
                        "harmful_cue_preserved": "",
                        "stance_preserved": "",
                        "context_sufficient_for_review": "",
                        "misleading_after_reduction": "",
                        "obvious_privacy_risk_remaining": "",
                        "review_notes": "",
                    }
                )


def _write_result_readme(output_dir: Path, config: dict[str, Any], metrics_by_variant: dict[str, dict[str, Any]]) -> None:
    run_name = output_dir.name
    classifier = config.get("classifier_resolved", config.get("classifier"))
    rows = next(iter(metrics_by_variant.values()), {}).get("row_count", 0)
    command = _result_command(config)
    lines = [
        f"# {run_name}",
        "",
        f"Rows: `{rows}`",
        f"Dataset: `{config.get('dataset')}`",
        f"Classifier: `{classifier}`",
        "",
        "Run command:",
        "",
        "```bash",
        command,
        "```",
        "",
        "Contents:",
        "",
        "- `main_table.md`: compact paper-style ablation table.",
        "- `variant_metrics.csv`: flat aggregate metrics for analysis.",
        "- `variant_metrics.json`: full aggregate metric dictionary.",
        "- `context_breakdown.csv`: grouped utility/context metrics.",
        "- `privacy_breakdown.csv`: PII and identifier residual metrics.",
        "- `dataset_summary.json`: sampled dataset summary.",
        "- `config.json`: run configuration, including classifier label mapping.",
        "- `qualitative_examples.md`: synthetic example table.",
        "",
        "Full transformed rows are stored outside the paper folder under:",
        "",
        "```text",
        f"{config.get('full_output_dir', 'data/outputs')}/{run_name}/release_rows.jsonl",
        "```",
        "",
    ]
    if int(config.get("manual_review_sample_size", 0) or 0) > 0:
        lines.insert(
            lines.index("- `dataset_summary.json`: sampled dataset summary."),
            "- `manual_review_sample.csv`: legacy transformed-text review sheet, if explicitly enabled.",
        )
    output_dir.joinpath("README.md").write_text("\n".join(lines), encoding="utf-8")


def _result_command(config: dict[str, Any]) -> str:
    parts = [
        "micromamba run -n icem-research python scripts/run_icem_experiment.py",
        f"  --dataset {config.get('dataset')}",
    ]
    if config.get("sample_size") is not None:
        parts.append(f"  --sample-size {config['sample_size']}")
    if config.get("classifier") and config.get("classifier") != "tfidf":
        parts.append(f"  --classifier {config['classifier']}")
    if config.get("classifier_batch_size") != 16:
        parts.append(f"  --classifier-batch-size {config['classifier_batch_size']}")
    if config.get("classifier_device") != "auto":
        parts.append(f"  --classifier-device {config['classifier_device']}")
    if config.get("include_hatecheck"):
        parts.append("  --include-hatecheck")
    if not config.get("inject_synthetic_pii", True):
        parts.append("  --no-inject-synthetic-pii")
    parts.extend(
        [
            f"  --output-dir {config.get('output_dir')}",
            "  --overwrite",
        ]
    )
    return " \\\n".join(parts)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _serializable_config(
    args: argparse.Namespace,
    variants: tuple[str, ...],
    classifier_name: str,
    classifier_details: str,
) -> dict[str, Any]:
    config = vars(args).copy()
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value)
    config["variants"] = list(variants)
    config["classifier_resolved"] = classifier_name
    config["classifier_details"] = classifier_details
    return config


if __name__ == "__main__":
    raise SystemExit(main())
