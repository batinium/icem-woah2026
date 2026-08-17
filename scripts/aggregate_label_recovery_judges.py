"""Aggregate label/target-recovery results across independent judge models.

Reads the primary judge summary (run 026 base, qwen/qwen3.5-9b) plus every
by_model/<slug>/summary.json, and reports whether the HEADLINE finding holds per
judge: (a) I-CEM hate-label fidelity > token-matched window, significant; (b) I-CEM
target recovery > matched window, significant; (c) importance-only corrupts >=50%
of non-hate as hate. Writes a paper-ready cross-judge table.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/paper_woah_2026/results/026_label_recovery_judge"
VARIANTS = ["importance_only", "importance_window_r2", "window_r3_matched", "icem_context"]
SHORT = {"importance_only": "imp-only", "importance_window_r2": "win r=2",
         "window_r3_matched": "win r=3 (matched)", "icem_context": "I-CEM"}


def summaries():
    out = []
    base = BASE / "summary.json"
    if base.exists():
        out.append(json.loads(base.read_text()))
    for d in sorted((BASE / "by_model").glob("*/summary.json")) if (BASE / "by_model").exists() else []:
        out.append(json.loads(d.read_text()))
    return out


def main():
    sums = summaries()
    if not sums:
        print("no summaries found — run the judges + analyze first")
        return
    lines = ["# Cross-judge label/target recoverability (run 026 multi-judge)", ""]
    lines.append(f"{len(sums)} independent judges. Headline = I-CEM significantly beats the "
                 "token-matched window on label fidelity AND target recovery.\n")

    # hate fidelity table
    lines.append("## Hate-label fidelity (%) by judge\n")
    hdr = "| Judge | " + " | ".join(SHORT[v] for v in VARIANTS) + " | I-CEM vs matched (p) |"
    lines += [hdr, "| " + " --- |" * (len(VARIANTS) + 2)]
    for s in sums:
        fid = s["hate_fidelity"]["variants"]
        row = [s["model"]] + [f'{fid[v]["fidelity_pct"]}' if v in fid else "-" for v in VARIANTS]
        p = s["hate_fidelity"]["mcnemar"]["icem_vs_matched_window"].get("p", "?")
        sig = "*" if isinstance(p, (int, float)) and p < 0.05 else ""
        lines.append("| " + " | ".join(row) + f" | {p}{sig} |")

    # target recovery table
    lines.append("\n## Target recovery (%) by judge\n")
    lines += [hdr, "| " + " --- |" * (len(VARIANTS) + 2)]
    for s in sums:
        tr = s["target_recovery"]
        row = [s["model"]] + [f'{tr[v]["recovered_vs_raw_pct"]}' if v in tr else "-" for v in VARIANTS]
        p = tr.get("mcnemar_icem_vs_matched_window", {}).get("p", "?")
        sig = "*" if isinstance(p, (int, float)) and p < 0.05 else ""
        lines.append("| " + " | ".join(row) + f" | {p}{sig} |")

    # counterspeech-misread harm
    lines.append("\n## Non-hate misread as hate (NO->YES %) by judge\n")
    lines += ["| Judge | " + " | ".join(SHORT[v] for v in VARIANTS) + " |",
              "| " + " --- |" * (len(VARIANTS) + 1)]
    for s in sums:
        fl = s.get("hate_flip_detail", {})
        row = [s["model"]] + [f'{fl[v]["NO_to_YES_pct"]}' if v in fl else "-" for v in VARIANTS]
        lines.append("| " + " | ".join(row) + " |")

    # verdict
    lines.append("\n## Headline replication verdict\n")
    for s in sums:
        pf = s["hate_fidelity"]["mcnemar"]["icem_vs_matched_window"].get("p", 1)
        pt = s["target_recovery"].get("mcnemar_icem_vs_matched_window", {}).get("p", 1)
        fid = s["hate_fidelity"]["variants"]
        icem_gt = fid.get("icem_context", {}).get("fidelity_pct", 0) > fid.get("window_r3_matched", {}).get("fidelity_pct", 0)
        ok_fid = icem_gt and isinstance(pf, (int, float)) and pf < 0.05
        ok_tgt = isinstance(pt, (int, float)) and pt < 0.05
        lines.append(f"- **{s['model']}**: label-fidelity win {'YES' if ok_fid else 'directional'} "
                     f"(p={pf}); target-recovery win {'YES' if ok_tgt else 'directional'} (p={pt}).")

    out = BASE / "multi_judge_summary.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
