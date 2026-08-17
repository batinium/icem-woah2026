#!/usr/bin/env python3
"""Summarize a completed compact author manual-audit CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = (
    "importance_only_context_sufficient",
    "importance_window_context_sufficient",
    "icem_context_sufficient",
    "icem_not_more_misleading_than_window",
    "obvious_privacy_risk_remaining",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    with args.input_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    summary = summarize(rows)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    args.output_md.write_text(to_markdown(summary), encoding="utf-8")
    return 0


def summarize(rows: list[dict[str, str]]) -> dict[str, object]:
    completed = [row for row in rows if any(row.get(field, "").strip() for field in FIELDS)]
    by_field = {field: counts(completed, field) for field in FIELDS}
    best_context = counts(completed, "best_context_variant")
    return {
        "row_count": len(rows),
        "completed_row_count": len(completed),
        "field_counts": by_field,
        "best_context_counts": best_context,
        "notes_nonempty_count": sum(1 for row in completed if row.get("review_notes", "").strip()),
    }


def counts(rows: list[dict[str, str]], field: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for row in rows:
        value = row.get(field, "").strip().lower() or "blank"
        values[value] = values.get(value, 0) + 1
    return dict(sorted(values.items()))


def pct(part: int, total: int) -> str:
    if total == 0:
        return "n/a"
    return f"{100 * part / total:.1f}%"


def to_markdown(summary: dict[str, object]) -> str:
    completed = int(summary["completed_row_count"])
    lines = [
        "# Author Manual Audit Summary",
        "",
        f"Completed rows: {completed}/{summary['row_count']}",
        "",
        "## Context Sufficiency",
        "",
        "| Field | yes | no | uncertain |",
        "| --- | ---: | ---: | ---: |",
    ]
    field_counts = summary["field_counts"]
    assert isinstance(field_counts, dict)
    for field in (
        "importance_only_context_sufficient",
        "importance_window_context_sufficient",
        "icem_context_sufficient",
    ):
        counts_for_field = field_counts[field]
        lines.append(
            "| "
            + field
            + " | "
            + f"{counts_for_field.get('yes', 0)} ({pct(counts_for_field.get('yes', 0), completed)})"
            + " | "
            + f"{counts_for_field.get('no', 0)} ({pct(counts_for_field.get('no', 0), completed)})"
            + " | "
            + f"{counts_for_field.get('uncertain', 0)} ({pct(counts_for_field.get('uncertain', 0), completed)})"
            + " |"
        )
    lines.extend(
        [
            "",
            "## Best Context Variant",
            "",
            "| Variant | Count |",
            "| --- | ---: |",
        ]
    )
    best_counts = summary["best_context_counts"]
    assert isinstance(best_counts, dict)
    for value, count in best_counts.items():
        lines.append(f"| {value} | {count} |")
    lines.extend(
        [
            "",
            "## Safety Checks",
            "",
            f"- I-CEM not more misleading than window: {field_counts['icem_not_more_misleading_than_window']}",
            f"- Obvious privacy risk remaining: {field_counts['obvious_privacy_risk_remaining']}",
            f"- Rows with notes: {summary['notes_nonempty_count']}",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
