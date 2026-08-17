"""Aggregate multi-seed HateXplain runs into mean +/- std for the key Table 1
columns. Reads each run's saved ``variant_metrics.json`` so it adds no compute;
it only summarizes runs that already exist on disk.

Usage:
    micromamba run -n icem-research python scripts/summarize_seed_variance.py \
        --runs docs/paper_woah_2026/results/005_hatexplain_dehatebert_5000 \
               docs/paper_woah_2026/results/016_hatexplain_seed7_5000 \
               docs/paper_woah_2026/results/017_hatexplain_seed23_5000 \
               docs/paper_woah_2026/results/018_hatexplain_seed41_5000 \
        --out docs/paper_woah_2026/seed_variance_summary.md
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

VARIANTS = ["importance_only", "importance_window", "icem_context"]
VARIANT_LABELS = {
    "importance_only": "Importance only",
    "importance_window": "Importance window (r=2)",
    "icem_context": "I-CEM",
}
# (json key, display label, scale to apply to reach a percentage/point value)
COLUMNS = [
    ("positive_f1", "F1+", 1.0),
    ("prediction_flip_rate_raw", "Flip%", 100.0),
    ("retained_token_pct", "Tok%", 1.0),
    ("target_cue_preservation", "Target%", 100.0),
    ("stance_harm_relation_preservation", "Stance%", 100.0),
    ("rationale_overlap", "Rationale%", 100.0),
]


def load_run(run_dir: Path) -> dict:
    metrics_path = run_dir / "variant_metrics.json"
    with metrics_path.open() as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="*", default=None,
                        help="Optional seed labels, parallel to --runs.")
    parser.add_argument("--drop-rationale", action="store_true",
                        help="Exclude the rationale-overlap column. Use this for "
                             "PII-injected runs, where the rationale mask is "
                             "misaligned by the injected prefix; rationale overlap "
                             "is only valid on no-PII runs.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    columns = [c for c in COLUMNS if not (args.drop_rationale and c[0] == "rationale_overlap")]

    runs = [load_run(Path(r)) for r in args.runs]
    seed_labels = args.seeds or [Path(r).name for r in args.runs]

    lines: list[str] = []
    lines.append("# Multi-seed variance summary")
    lines.append("")
    lines.append(
        f"Aggregated over {len(runs)} HateXplain runs "
        f"({', '.join(seed_labels)}), 5,000 rows each, identical config except "
        "the sampling seed. Values are mean +/- sample std across seeds. "
        "F1+ is the positive-class F1; Flip/Tok/Target/Stance are percentages."
    )
    lines.append("")

    header = "| Variant | " + " | ".join(label for _, label, _ in columns) + " |"
    sep = "| --- | " + " | ".join("---" for _ in columns) + " |"
    lines.append(header)
    lines.append(sep)

    for variant in VARIANTS:
        cells = [VARIANT_LABELS[variant]]
        for key, _label, scale in columns:
            vals = [run[variant][key] * scale for run in runs]
            mean = statistics.mean(vals)
            std = statistics.stdev(vals) if len(vals) > 1 else 0.0
            if key == "positive_f1":
                cells.append(f"{mean:.3f} +/- {std:.3f}")
            else:
                cells.append(f"{mean:.1f} +/- {std:.1f}")
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("## Per-seed raw values")
    lines.append("")
    for variant in VARIANTS:
        lines.append(f"### {VARIANT_LABELS[variant]}")
        lines.append("")
        lines.append("| Seed | " + " | ".join(label for _, label, _ in columns) + " |")
        lines.append("| --- | " + " | ".join("---" for _ in columns) + " |")
        for seed_label, run in zip(seed_labels, runs):
            cells = [seed_label]
            for key, _label, scale in columns:
                val = run[variant][key] * scale
                cells.append(f"{val:.3f}" if key == "positive_f1" else f"{val:.1f}")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
