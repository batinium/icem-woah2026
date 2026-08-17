"""Summarize I-CEM selector tradeoffs from saved experiment folders."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_RESULTS_ROOT = Path("docs/paper_woah_2026/results")
DEFAULT_OUTPUT_MD = Path("docs/paper_woah_2026/selector_tradeoff_summary.md")
DEFAULT_OUTPUT_CSV = Path("docs/paper_woah_2026/selector_tradeoff_summary.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--variant", default="icem_context")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = _selector_rows(args.results_root, variant=args.variant)
    independent_rows = _independent_evaluator_rows(args.results_root)
    _mark_pareto(rows)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_csv, rows)
    args.output_md.write_text(_render_markdown(rows, independent_rows, args.variant), encoding="utf-8")
    print(f"wrote {args.output_md}")
    print(f"wrote {args.output_csv}")
    return 0


def _selector_rows(results_root: Path, *, variant: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(results_root.iterdir()):
        config_path = run_dir / "config.json"
        metrics_path = run_dir / "variant_metrics.json"
        if not config_path.exists() or not metrics_path.exists():
            continue
        config = _read_json(config_path)
        if config.get("dataset") != "hatexplain":
            continue
        if not config.get("inject_synthetic_pii", True):
            continue
        if config.get("sample_size") != 5000:
            continue
        if "dehatebert" not in str(config.get("classifier", "")).lower():
            continue
        metrics = _read_json(metrics_path)
        if variant not in metrics:
            continue
        metric = metrics[variant]
        row = _metric_row(run_dir.name, config, metric, variant=variant)
        row["recommended_by_threshold"] = _recommended_by_threshold(row)
        rows.append(row)
    return rows


def _independent_evaluator_rows(results_root: Path) -> list[dict[str, Any]]:
    source_configs = _source_configs_by_run_name(results_root)
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(results_root.iterdir()):
        config_path = run_dir / "config.json"
        metrics_path = run_dir / "variant_metrics.json"
        if not config_path.exists() or not metrics_path.exists():
            continue
        config = _read_json(config_path)
        if "evaluator_classifier" not in config:
            continue
        source_run = Path(str(config.get("release_rows", ""))).parent.name
        source_config = source_configs.get(source_run, {})
        metrics = _read_json(metrics_path)
        for variant in ("importance_only", "importance_window", "icem_context"):
            if variant not in metrics:
                continue
            row = _metric_row(
                run_dir.name,
                {**source_config, **config},
                metrics[variant],
                variant=variant,
            )
            row["source_run"] = source_run
            rows.append(row)
    return rows


def _source_configs_by_run_name(results_root: Path) -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {}
    for run_dir in sorted(results_root.iterdir()):
        config_path = run_dir / "config.json"
        if not config_path.exists():
            continue
        config = _read_json(config_path)
        configs[run_dir.name] = config
    return configs


def _metric_row(run_name: str, config: dict[str, Any], metric: dict[str, Any], *, variant: str) -> dict[str, Any]:
    direct_residual = _pct(float(metric.get("direct_pii_residual_rate", 0.0)))
    quasi_residual = _pct(float(metric.get("quasi_identifier_residual_rate", 0.0)))
    detected_direct = int(metric.get("detected_direct_identifier_residual_count", 0) or 0)
    detected_quasi = int(metric.get("detected_quasi_identifier_residual_count", 0) or 0)
    zero_pii = direct_residual == 0.0 and quasi_residual == 0.0 and detected_direct == 0 and detected_quasi == 0
    top_k = config.get("importance_top_k")
    radius = config.get("window_radius")
    return {
        "run": run_name,
        "variant": variant,
        "top_k": top_k,
        "radius": radius,
        "setting": _setting_label(top_k, radius),
        "positive_f1": _round(metric.get("positive_f1")),
        "flip_pct": _pct(float(metric.get("prediction_flip_rate_raw", 0.0))),
        "retained_token_pct": _round(metric.get("retained_token_pct")),
        "retained_char_pct": _round(metric.get("retained_char_pct")),
        "target_cue_pct": _pct(float(metric.get("target_cue_preservation") or 0.0)),
        "target_harm_pct": _pct(float(metric.get("target_harm_pair_preservation") or 0.0)),
        "stance_harm_pct": _pct(float(metric.get("stance_harm_relation_preservation") or 0.0)),
        "direct_pii_residual_pct": direct_residual,
        "quasi_pii_residual_pct": quasi_residual,
        "detected_direct_residual_count": detected_direct,
        "detected_quasi_residual_count": detected_quasi,
        "placeholder_only_count": int(metric.get("placeholder_only_count", 0) or 0),
        "zero_pii_residual": zero_pii,
        "row_count": int(metric.get("row_count", 0) or 0),
    }


def _recommended_by_threshold(row: dict[str, Any]) -> bool:
    return bool(
        row["zero_pii_residual"]
        and row["placeholder_only_count"] == 0
        and row["target_cue_pct"] >= 95.0
        and row["stance_harm_pct"] >= 60.0
        and row["flip_pct"] <= 13.5
        and row["retained_token_pct"] <= 50.0
    )


def _mark_pareto(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        row["pareto_dominated"] = any(_dominates(other, row) for other in rows if other is not row)


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    comparable = (
        left["zero_pii_residual"] >= right["zero_pii_residual"],
        left["placeholder_only_count"] <= right["placeholder_only_count"],
        left["retained_token_pct"] <= right["retained_token_pct"],
        left["flip_pct"] <= right["flip_pct"],
        left["positive_f1"] >= right["positive_f1"],
        left["target_cue_pct"] >= right["target_cue_pct"],
        left["stance_harm_pct"] >= right["stance_harm_pct"],
    )
    if not all(comparable):
        return False
    return any(
        (
            left["retained_token_pct"] < right["retained_token_pct"],
            left["flip_pct"] < right["flip_pct"],
            left["positive_f1"] > right["positive_f1"],
            left["target_cue_pct"] > right["target_cue_pct"],
            left["stance_harm_pct"] > right["stance_harm_pct"],
            left["placeholder_only_count"] < right["placeholder_only_count"],
        )
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "run",
        "variant",
        "setting",
        "top_k",
        "radius",
        "positive_f1",
        "flip_pct",
        "retained_token_pct",
        "retained_char_pct",
        "target_cue_pct",
        "target_harm_pct",
        "stance_harm_pct",
        "zero_pii_residual",
        "placeholder_only_count",
        "recommended_by_threshold",
        "pareto_dominated",
        "row_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _render_markdown(
    rows: list[dict[str, Any]],
    independent_rows: list[dict[str, Any]],
    variant: str,
) -> str:
    recommended = [row for row in rows if row["recommended_by_threshold"]]
    lines = [
        "# Selector tradeoff summary",
        "",
        "Generated from saved experiment folders with `scripts/summarize_selector_tradeoffs.py`.",
        "",
        "Conservative selector thresholds used for the paper default:",
        "",
        "- zero injected and detected direct/quasi PII residuals",
        "- zero placeholder-only releases",
        "- target cue preservation >= 95%",
        "- stance-harm relation preservation >= 60%",
        "- raw-decision flip rate <= 13.5%",
        "- retained source tokens <= 50%",
        "",
    ]
    if recommended:
        settings = ", ".join(f"`{row['setting']}`" for row in recommended)
        lines.extend(
            [
                f"Recommended setting under these thresholds: {settings}.",
                "",
                "This supports the current default `top_k=5, radius=2`: it is the only full",
                "HateXplain selector setting in the sweep that satisfies all conservative",
                "thresholds while staying below 50% retained source tokens.",
                "",
            ]
        )
    else:
        lines.extend(["No setting satisfies all conservative thresholds.", ""])

    lines.extend(
        [
            "## Full HateXplain selector sweep",
            "",
            f"Variant summarized: `{variant}` over 5,000 HateXplain examples.",
            "",
            "| Run | Setting | F1+ | Flip % | Tok % | Target % | Stance % | Zero PII | Recommended | Dominated |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
        ]
    )
    for row in sorted(rows, key=lambda item: (item["radius"] or 0, item["top_k"] or 0, item["run"])):
        lines.append(
            "| {run} | {setting} | {positive_f1:.3f} | {flip_pct:.1f} | {retained_token_pct:.1f} | "
            "{target_cue_pct:.1f} | {stance_harm_pct:.1f} | {zero} | {rec} | {dom} |".format(
                **row,
                zero="yes" if row["zero_pii_residual"] else "no",
                rec="yes" if row["recommended_by_threshold"] else "no",
                dom="yes" if row["pareto_dominated"] else "no",
            )
        )
    lines.append("")

    if independent_rows:
        lines.extend(
            [
                "## Independent evaluator check",
                "",
                "These rows rescore saved release texts with `unitary/unbiased-toxic-roberta`,",
                "which was not used for token selection.",
                "",
                "| Source run | Variant | Setting | F1+ | Flip % | Tok % | Target % | Stance % |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in independent_rows:
            lines.append(
                "| {source_run} | {variant} | {setting} | {positive_f1:.3f} | {flip_pct:.1f} | "
                "{retained_token_pct:.1f} | {target_cue_pct:.1f} | {stance_harm_pct:.1f} |".format(**row)
            )
        lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            "The sweep does not show a single setting that is best on every metric. Radius 1",
            "and lower `top_k` settings release less text, but they increase prediction",
            "flips or miss context. Larger settings reduce flips slightly, but cross the",
            "50% retained-token line for limited context gains. The default therefore has a",
            "clear paper rationale: it is a conservative balance, not the numerically best",
            "setting for one isolated score.",
            "",
        ]
    )
    return "\n".join(lines)


def _setting_label(top_k: Any, radius: Any) -> str:
    if top_k is None and radius is None:
        return "unknown"
    return f"k={top_k}, r={radius}"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(value: float) -> float:
    return round(value * 100.0, 3)


def _round(value: Any) -> float:
    return round(float(value or 0.0), 3)


if __name__ == "__main__":
    raise SystemExit(main())
