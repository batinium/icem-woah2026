"""Report-generation placeholder for paper tables."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join([header_line, divider, *body])


def write_metrics_csv(path: Path, metrics_by_variant: dict[str, dict[str, Any]]) -> None:
    keys: list[str] = []
    for metrics in metrics_by_variant.values():
        for key in metrics:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variant", *keys])
        writer.writeheader()
        for variant, metrics in metrics_by_variant.items():
            writer.writerow({"variant": variant, **metrics})


def main_result_table(metrics_by_variant: dict[str, dict[str, Any]]) -> str:
    headers = [
        "Variant",
        "F1+",
        "Flip%",
        "Tok%",
        "Char%",
        "PII resid%",
        "Target%",
        "Stance%",
    ]
    rows: list[list[object]] = []
    for variant, metrics in metrics_by_variant.items():
        rows.append(
            [
                variant,
                _fmt(metrics.get("positive_f1")),
                _pct(metrics.get("prediction_flip_rate_raw")),
                _fmt(metrics.get("retained_token_pct")),
                _fmt(metrics.get("retained_char_pct")),
                _pct(metrics.get("direct_pii_residual_rate")),
                _pct(metrics.get("target_cue_preservation")),
                _pct(metrics.get("stance_harm_relation_preservation")),
            ]
        )
    return markdown_table(headers, rows) + "\n"


def qualitative_examples_table() -> str:
    warning = (
        "**Content warning:** examples below are synthetic schematics illustrating "
        "method behavior, not real dataset posts. They contain hate-adjacent language "
        "for demonstration purposes only.\n\n"
    )
    headers = ["Phenomenon", "Synthetic Source", "Importance Only", "I-CEM", "Note"]
    rows = [
        [
            "Negation",
            "I do not agree with people saying Muslims should be banned.",
            "Muslims banned",
            "do not agree with people saying Muslims should be banned",
            "stance recovered",
        ],
        [
            "Quotation",
            'She quoted "Jews are vermin" and condemned it.',
            "Jews vermin",
            'quoted "Jews are vermin" and condemned',
            "reporting context retained",
        ],
        [
            "Synthetic PII",
            "Alex Mercer from Riverton High wrote: immigrants should be kicked out.",
            "immigrants kicked out",
            "immigrants should be kicked out",
            "identifier context removed",
        ],
    ]
    return warning + markdown_table(headers, rows) + "\n"


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _pct(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value) * 100:.1f}" if 0 <= float(value) <= 1 else f"{float(value):.1f}"
