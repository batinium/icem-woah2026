"""Create a paper-facing HateCheck context summary from context_breakdown.csv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("docs/paper_woah_2026/results/003_hatecheck_dehatebert_full/context_breakdown.csv")
DEFAULT_OUTPUT = Path("docs/paper_woah_2026/hatecheck_context_summary.md")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = _read_rows(args.input_csv)
    args.output_md.write_text(_render(rows), encoding="utf-8")
    print(f"wrote {args.output_md}")
    return 0


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [_convert_row(row) for row in reader]


def _convert_row(row: dict[str, str]) -> dict[str, Any]:
    converted: dict[str, Any] = {}
    for key, value in row.items():
        if key in {"group_type", "group", "variant"}:
            converted[key] = value
        elif value == "":
            converted[key] = None
        elif key == "row_count":
            converted[key] = int(value)
        else:
            converted[key] = float(value)
    return converted


def _render(rows: list[dict[str, Any]]) -> str:
    by_key = {(row["group_type"], row["group"], row["variant"]): row for row in rows}
    dataset_rows = [row for row in rows if row["group_type"] == "dataset"]
    functionality_groups = sorted({row["group"] for row in rows if row["group_type"] == "functionality"})
    comparisons = _comparisons(by_key, functionality_groups)

    lines = [
        "# HateCheck context summary",
        "",
        "Generated from `results/003_hatecheck_dehatebert_full/context_breakdown.csv`",
        "with `scripts/summarize_hatecheck_context.py`.",
        "",
        "## Dataset-level results",
        "",
        "| Variant | F1+ | Flip % | Tok % | Target % | Negation % | Quote % | Counter % | Target-harm % | Stance-harm % |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant in ("importance_only", "importance_window", "icem_context"):
        row = next(item for item in dataset_rows if item["variant"] == variant)
        lines.append(
            "| {variant} | {positive_f1:.3f} | {flip:.1f} | {tok:.1f} | {target:.1f} | "
            "{neg:.1f} | {quote:.1f} | {counter:.1f} | {target_harm:.1f} | {stance_harm:.1f} |".format(
                variant=variant,
                positive_f1=row["positive_f1"],
                flip=_pct(row["prediction_flip_rate_raw"]),
                tok=row["retained_token_pct"],
                target=_pct(row["target_cue_preservation"]),
                neg=_pct(row["negation_cue_preservation"]),
                quote=_pct(row["quotation_cue_preservation"]),
                counter=_pct(row["counterspeech_cue_preservation"]),
                target_harm=_pct(row["target_harm_pair_preservation"]),
                stance_harm=_pct(row["stance_harm_relation_preservation"]),
            )
        )

    lines.extend(
        [
            "",
            "## Functionality-level comparison",
            "",
            _comparison_sentence(comparisons, "importance_only"),
            _comparison_sentence(comparisons, "importance_window"),
            "",
            "## Largest stance gains over fixed windows",
            "",
            "| Functionality | n | Window stance % | I-CEM stance % | Delta | Token cost | Flip delta |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in _top_deltas(comparisons["importance_window"], "stance_harm_delta", limit=8):
        lines.append(_delta_row(item))

    lines.extend(
        [
            "",
            "## Largest flip reductions over fixed windows",
            "",
            "| Functionality | n | Window stance % | I-CEM stance % | Delta | Token cost | Flip delta |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in sorted(comparisons["importance_window"], key=lambda row: row["flip_delta"])[:8]:
        lines.append(_delta_row(item))

    lines.extend(
        [
            "",
            "## Paper interpretation",
            "",
            "HateCheck is a stress test for context rather than a strong classifier-utility",
            "benchmark here: DeHateBERT has low positive F1 on raw HateCheck. The useful",
            "claim is narrower and stronger: I-CEM recovers target, negation, quotation,",
            "counter-speech, and stance context that importance-only and fixed-window",
            "releases frequently drop. The cost is higher retained-token exposure than",
            "fixed windows, which should be reported as the privacy-utility tradeoff.",
            "",
        ]
    )
    return "\n".join(lines)


def _comparisons(
    by_key: dict[tuple[str, str, str], dict[str, Any]],
    groups: list[str],
) -> dict[str, list[dict[str, Any]]]:
    output = {"importance_only": [], "importance_window": []}
    for baseline in output:
        for group in groups:
            base = by_key.get(("functionality", group, baseline))
            icem = by_key.get(("functionality", group, "icem_context"))
            if not base or not icem:
                continue
            output[baseline].append(
                {
                    "group": group,
                    "row_count": icem["row_count"],
                    "base_stance": base.get("stance_harm_relation_preservation"),
                    "icem_stance": icem.get("stance_harm_relation_preservation"),
                    "stance_harm_delta": _delta(
                        icem.get("stance_harm_relation_preservation"),
                        base.get("stance_harm_relation_preservation"),
                        pct=True,
                    ),
                    "target_delta": _delta(
                        icem.get("target_cue_preservation"),
                        base.get("target_cue_preservation"),
                        pct=True,
                    ),
                    "negation_delta": _delta(
                        icem.get("negation_cue_preservation"),
                        base.get("negation_cue_preservation"),
                        pct=True,
                    ),
                    "quote_delta": _delta(
                        icem.get("quotation_cue_preservation"),
                        base.get("quotation_cue_preservation"),
                        pct=True,
                    ),
                    "flip_delta": _delta(
                        icem.get("prediction_flip_rate_raw"),
                        base.get("prediction_flip_rate_raw"),
                        pct=True,
                    ),
                    "token_cost": _delta(
                        icem.get("retained_token_pct"),
                        base.get("retained_token_pct"),
                        pct=False,
                    ),
                }
            )
    return output


def _comparison_sentence(comparisons: dict[str, list[dict[str, Any]]], baseline: str) -> str:
    items = comparisons[baseline]
    target = _count_positive(items, "target_delta")
    stance = _count_positive(items, "stance_harm_delta")
    negation = _count_positive(items, "negation_delta")
    quote = _count_positive(items, "quote_delta")
    flip = _count_negative(items, "flip_delta")
    return (
        f"Against `{baseline}`, I-CEM improves target preservation in {target[0]}/{target[1]} "
        f"functionality groups, stance-harm preservation in {stance[0]}/{stance[1]}, "
        f"negation preservation in {negation[0]}/{negation[1]}, quotation preservation in "
        f"{quote[0]}/{quote[1]}, and lowers raw-decision flips in {flip[0]}/{flip[1]} groups."
    )


def _top_deltas(items: list[dict[str, Any]], key: str, *, limit: int) -> list[dict[str, Any]]:
    available = [item for item in items if item[key] is not None]
    return sorted(available, key=lambda row: row[key], reverse=True)[:limit]


def _delta_row(item: dict[str, Any]) -> str:
    return (
        "| {group} | {row_count} | {base_stance} | {icem_stance} | {stance_delta} | {token_cost} | {flip_delta} |".format(
            group=item["group"],
            row_count=item["row_count"],
            base_stance=_fmt(item["base_stance"], pct=True),
            icem_stance=_fmt(item["icem_stance"], pct=True),
            stance_delta=_fmt_delta(item["stance_harm_delta"]),
            token_cost=_fmt_delta(item["token_cost"]),
            flip_delta=_fmt_delta(item["flip_delta"]),
        )
    )


def _count_positive(items: list[dict[str, Any]], key: str) -> tuple[int, int]:
    available = [item[key] for item in items if item[key] is not None]
    return sum(1 for value in available if value > 0), len(available)


def _count_negative(items: list[dict[str, Any]], key: str) -> tuple[int, int]:
    available = [item[key] for item in items if item[key] is not None]
    return sum(1 for value in available if value < 0), len(available)


def _delta(left: Any, right: Any, *, pct: bool) -> float | None:
    if left is None or right is None:
        return None
    scale = 100.0 if pct else 1.0
    return round((float(left) - float(right)) * scale, 3)


def _pct(value: Any) -> float:
    if value is None:
        return 0.0
    return round(float(value) * 100.0, 1)


def _fmt(value: Any, *, pct: bool = False) -> str:
    if value is None:
        return "n/a"
    if pct:
        value = float(value) * 100.0
    return f"{float(value):.1f}"


def _fmt_delta(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):+.1f}"


if __name__ == "__main__":
    raise SystemExit(main())
