#!/usr/bin/env python3
"""Summarize a completed I-CEM manual-review CSV."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any


REVIEW_FIELDS = (
    "label_preserved",
    "target_preserved",
    "harmful_cue_preserved",
    "stance_preserved",
    "context_sufficient_for_review",
    "misleading_after_reduction",
    "obvious_privacy_risk_remaining",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize manual review decisions by variant.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = read_rows(args.input_csv)
    summary = summarize(rows)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown = render_markdown(summary)
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown, encoding="utf-8")
    else:
        print(markdown)
    return 0


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
    missing = [
        {"review_id": row.get("review_id"), "variant": row.get("variant"), "field": field}
        for row in rows
        for field in REVIEW_FIELDS
        if not row.get(field, "").strip()
    ]
    variants = sorted({row.get("variant", "") for row in rows if row.get("variant")})
    by_variant: dict[str, Any] = {}
    for variant in variants:
        subset = [row for row in rows if row.get("variant") == variant]
        field_counts = {field: dict(Counter(normalize(row.get(field, "")) for row in subset)) for field in REVIEW_FIELDS}
        by_variant[variant] = {
            "rows": len(subset),
            "field_counts": field_counts,
            "rates": {
                "label_preserved_yes": rate(field_counts["label_preserved"], "yes"),
                "target_preserved_yes": rate_excluding_na(field_counts["target_preserved"], "yes"),
                "harmful_cue_preserved_yes": rate_excluding_na(field_counts["harmful_cue_preserved"], "yes"),
                "stance_preserved_yes": rate_excluding_na(field_counts["stance_preserved"], "yes"),
                "context_sufficient_yes": rate(field_counts["context_sufficient_for_review"], "yes"),
                "misleading_yes": rate(field_counts["misleading_after_reduction"], "yes"),
                "privacy_risk_yes": rate(field_counts["obvious_privacy_risk_remaining"], "yes"),
            },
        }
    return {
        "rows": len(rows),
        "variants": variants,
        "missing_required_fields": missing,
        "missing_required_count": len(missing),
        "notes_nonempty_count": sum(1 for row in rows if row.get("review_notes", "").strip()),
        "label_counts": dict(Counter(row.get("label", "") for row in rows)),
        "by_variant": by_variant,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Manual Review Summary",
        "",
        f"Rows: `{summary['rows']}`",
        f"Missing required fields: `{summary['missing_required_count']}`",
        f"Rows with notes: `{summary['notes_nonempty_count']}`",
        "",
        "| Variant | Label yes% | Target yes% | Harm yes% | Stance yes% | Context yes% | Misleading yes% | Privacy risk yes% |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant, payload in summary["by_variant"].items():
        rates = payload["rates"]
        lines.append(
            "| "
            + " | ".join(
                [
                    variant,
                    fmt_pct(rates["label_preserved_yes"]),
                    fmt_pct(rates["target_preserved_yes"]),
                    fmt_pct(rates["harmful_cue_preserved_yes"]),
                    fmt_pct(rates["stance_preserved_yes"]),
                    fmt_pct(rates["context_sufficient_yes"]),
                    fmt_pct(rates["misleading_yes"]),
                    fmt_pct(rates["privacy_risk_yes"]),
                ]
            )
            + " |"
        )
    if summary["missing_required_fields"]:
        lines.extend(["", "## Missing Fields", ""])
        for item in summary["missing_required_fields"][:50]:
            lines.append(f"- review_id={item['review_id']} variant={item['variant']} field={item['field']}")
        if len(summary["missing_required_fields"]) > 50:
            lines.append(f"- ... {len(summary['missing_required_fields']) - 50} more")
    return "\n".join(lines) + "\n"


def normalize(value: str) -> str:
    cleaned = value.strip().lower()
    return cleaned or "missing"


def rate(counts: dict[str, int], value: str) -> float | None:
    total = sum(counts.values())
    if total == 0:
        return None
    return counts.get(value, 0) / total


def rate_excluding_na(counts: dict[str, int], value: str) -> float | None:
    total = sum(count for key, count in counts.items() if key != "na")
    if total == 0:
        return None
    return counts.get(value, 0) / total


def fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}"


if __name__ == "__main__":
    raise SystemExit(main())
