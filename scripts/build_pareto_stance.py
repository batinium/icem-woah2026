"""Build the exposure-vs-stance Pareto frontier for the fixed-window family vs I-CEM.

Reuses the independent stance judge (run 025) for the existing points and judges the
window r=1 excerpts (run 027) to fill the gap between importance-only and r=2.
"""
from __future__ import annotations

import json
from pathlib import Path

import importlib.util

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("stance_judge_mod", ROOT / "scripts/run_stance_llm_judge.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
judge, NONASSERT = _mod.judge, _mod.NONASSERT
S025 = ROOT / "docs/paper_woah_2026/results/025_stance_llm_judge/checkpoint.jsonl"
R027 = ROOT / "data/outputs/027_hatexplain_nopii_radius1_5000/release_rows.jsonl"
OUT = ROOT / "docs/paper_woah_2026/results/025_stance_llm_judge/pareto.json"


def main():
    recs = [json.loads(l) for l in S025.open()]
    ref = [r for r in recs if r["labels"].get("raw") in NONASSERT]
    ref_ids = {r["row_id"] for r in ref}

    r027 = {}
    with R027.open() as f:
        for line in f:
            r = json.loads(line)
            r027[r["row"]["row_id"]] = r["variants"]["importance_window"]["released_text"]

    # judge r=1 window excerpts on the same reference rows
    preserved = 0
    for r in ref:
        excerpt = r027.get(r["row_id"], "")
        preserved += int(judge(excerpt) in NONASSERT)
    r1_rate = round(100 * preserved / len(ref), 1)

    # token% for r=1 window from run 027 variant_metrics
    vm = (ROOT / "docs/paper_woah_2026/results/027_hatexplain_nopii_radius1_5000/variant_metrics.csv").read_text().splitlines()
    hdr = vm[0].split(","); ti = hdr.index("retained_token_pct")
    r1_tok = None
    for line in vm[1:]:
        c = line.split(",")
        if c[0] == "importance_window":
            r1_tok = float(c[ti])

    points = {
        "importance_only_r0": {"tok": 14.3, "stance": 1.3, "family": "window"},
        "window_r1": {"tok": round(r1_tok, 1) if r1_tok else None, "stance": r1_rate, "family": "window"},
        "window_r2": {"tok": 49.7, "stance": 26.3, "family": "window"},
        "window_r3": {"tok": 59.7, "stance": 42.1, "family": "window"},
        "icem": {"tok": 57.5, "stance": 49.1, "family": "icem"},
    }
    # I-CEM vs the window frontier at I-CEM's token budget (linear interp between r2 and r3)
    t, s2, s3, t2, t3 = 57.5, 26.3, 42.1, 49.7, 59.7
    interp = s2 + (t - t2) / (t3 - t2) * (s3 - s2)
    points["icem_vs_window_frontier_at_57.5pct"] = {"window_interp_stance": round(interp, 1),
                                                     "icem_stance": 49.1,
                                                     "icem_above_frontier_pts": round(49.1 - interp, 1)}
    OUT.write_text(json.dumps(points, indent=2))
    print(json.dumps(points, indent=2))


if __name__ == "__main__":
    main()
