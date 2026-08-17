#!/usr/bin/env python3
"""Evaluate already-generated I-CEM release rows with another classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.icem.classifier import build_classifier
from scripts.icem.metrics import aggregate_variant_metrics
from scripts.icem.report import main_result_table, write_metrics_csv
from scripts.icem.schema import ReleaseResult, Row, Span


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score saved I-CEM release rows with a frozen evaluator.")
    parser.add_argument("--release-rows", type=Path, required=True)
    parser.add_argument("--classifier", required=True)
    parser.add_argument("--classifier-batch-size", type=int, default=64)
    parser.add_argument("--classifier-device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.output_dir.exists() and not args.overwrite:
        parser.error(f"output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows, results_by_variant = _read_release_rows(args.release_rows)
    classifier_result = build_classifier(
        args.classifier,
        rows,
        seed=args.seed,
        batch_size=args.classifier_batch_size,
        device=args.classifier_device,
    )
    classifier = classifier_result.classifier

    raw_scores = classifier.predict_proba([result.released_text for result in results_by_variant["raw"]])
    metrics_by_variant: dict[str, dict[str, Any]] = {}
    for variant, results in results_by_variant.items():
        scores = classifier.predict_proba([result.released_text for result in results])
        metrics_by_variant[variant] = aggregate_variant_metrics(
            rows,
            results,
            released_scores=scores,
            raw_scores=raw_scores,
        )

    config = {
        "release_rows": str(args.release_rows),
        "row_count": len(rows),
        "variants": list(results_by_variant),
        "evaluator_classifier": args.classifier,
        "classifier_batch_size": args.classifier_batch_size,
        "classifier_device": args.classifier_device,
        "classifier_resolved": classifier_result.name,
        "classifier_details": classifier_result.details,
        "seed": args.seed,
    }
    _write_json(args.output_dir / "config.json", config)
    _write_json(args.output_dir / "variant_metrics.json", metrics_by_variant)
    write_metrics_csv(args.output_dir / "variant_metrics.csv", metrics_by_variant)
    (args.output_dir / "main_table.md").write_text(main_result_table(metrics_by_variant), encoding="utf-8")
    _write_readme(args.output_dir, config)
    return 0


def _read_release_rows(path: Path) -> tuple[tuple[Row, ...], dict[str, list[ReleaseResult]]]:
    rows: list[Row] = []
    results_by_variant: dict[str, list[ReleaseResult]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            rows.append(_row_from_dict(payload["row"]))
            for variant, raw_result in payload["variants"].items():
                results_by_variant.setdefault(variant, []).append(_release_from_dict(raw_result))
    if "raw" not in results_by_variant:
        raise RuntimeError("release rows must include a raw variant")
    return tuple(rows), results_by_variant


def _row_from_dict(raw: dict[str, Any]) -> Row:
    return Row(
        row_id=str(raw["row_id"]),
        dataset=str(raw["dataset"]),
        split=str(raw["split"]),
        text=str(raw["text"]),
        label=raw.get("label"),
        binary_label=raw.get("binary_label"),
        target_groups=tuple(raw.get("target_groups") or ()),
        rationale_token_mask=tuple(raw["rationale_token_mask"]) if raw.get("rationale_token_mask") else None,
        functionality=raw.get("functionality"),
        metadata=raw.get("metadata") or {},
    )


def _release_from_dict(raw: dict[str, Any]) -> ReleaseResult:
    return ReleaseResult(
        row_id=str(raw["row_id"]),
        variant=str(raw["variant"]),
        source_text=str(raw["source_text"]),
        released_text=str(raw["released_text"]),
        kept_spans=tuple(_span_from_dict(span) for span in raw.get("kept_spans", ())),
        masked_spans=tuple(_span_from_dict(span) for span in raw.get("masked_spans", ())),
        warnings=tuple(raw.get("warnings", ())),
        metadata=raw.get("metadata") or {},
    )


def _span_from_dict(raw: dict[str, Any]) -> Span:
    return Span(
        start=int(raw["start"]),
        end=int(raw["end"]),
        label=str(raw["label"]),
        source=str(raw.get("source", "unknown")),
        score=float(raw.get("score", 1.0)),
        replacement=raw.get("replacement"),
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_readme(output_dir: Path, config: dict[str, Any]) -> None:
    lines = [
        f"# {output_dir.name}",
        "",
        "Cross-classifier evaluation of already-generated I-CEM release rows.",
        "",
        f"Release rows: `{config['release_rows']}`",
        f"Rows: `{config['row_count']}`",
        f"Evaluator: `{config['classifier_resolved']}`",
        "",
        "This run does not recompute token importance or release text. It scores",
        "the same released variants with an independent frozen evaluator to probe",
        "whether utility preservation is tied to the selector classifier.",
        "",
        "Contents:",
        "",
        "- `main_table.md`: compact evaluator table.",
        "- `variant_metrics.csv`: flat aggregate metrics.",
        "- `variant_metrics.json`: full metric dictionary.",
        "- `config.json`: evaluator configuration and label mapping.",
        "",
    ]
    output_dir.joinpath("README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
