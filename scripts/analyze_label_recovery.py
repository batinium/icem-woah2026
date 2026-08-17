"""Analyze the independent label/target recoverability judge (run 026)."""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Directory + model label overridable so the same metrics run on every judge.
# Usage: python analyze_label_recovery.py [result_dir] [model_label]
D = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/paper_woah_2026/results/026_label_recovery_judge"
MODEL_LABEL = sys.argv[2] if len(sys.argv) > 2 else "qwen/qwen3.5-9b (LAN-local)"
VARIANTS = ["importance_only", "importance_window_r2", "window_r3_matched", "icem_context"]
TOK = {"importance_only": 14.3, "importance_window_r2": 49.7, "window_r3_matched": 59.7, "icem_context": 57.5}


def toks(s):
    return {w for w in s.replace("/", " ").replace("-", " ").split() if len(w) > 3}


def target_match(pred, ref_tokens):
    p = toks(pred)
    return bool(p & ref_tokens) if ref_tokens else None


def mcnemar(rows, a, b, fn):
    b01 = sum(1 for r in rows if not fn(r, a) and fn(r, b))
    b10 = sum(1 for r in rows if fn(r, a) and not fn(r, b))
    n = b01 + b10
    if n == 0:
        return {"discordant": 0, "p": 1.0}
    k = min(b01, b10)
    p = min(1.0, sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n * 2)
    return {"discordant": n, f"{a}_only": b10, f"{b}_only": b01, "p": round(p, 4)}


def boot(rows, fn, var, iters=10000):
    vals = [1 if fn(r, var) else 0 for r in rows]
    n = len(vals); rng = random.Random(17); m = []
    for _ in range(iters):
        m.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    m.sort()
    return round(100 * m[int(.025 * iters)], 1), round(100 * m[int(.975 * iters)], 1)


def main():
    recs = [json.loads(l) for l in (D / "checkpoint.jsonl").open()]

    # --- hate-judgment fidelity vs the judge's read of the full raw text ---
    hate_ref = [r for r in recs if r["hate"].get("raw") in {"YES", "NO"}]

    def hate_fid(r, v):  # excerpt hate decision equals raw hate decision
        return r["hate"].get(v) == r["hate"]["raw"]

    out = {
        "model": MODEL_LABEL,
        "n_judged": len(recs),
        "hate_fidelity": {"denominator_raw_clear": len(hate_ref), "variants": {}, "mcnemar": {}},
        "target_recovery": {},
        "hate_flip_detail": {},
    }
    for v in VARIANTS:
        rows = [r for r in hate_ref if r["hate"].get(v) in {"YES", "NO", "UNCLEAR"}]
        fid = sum(hate_fid(r, v) for r in rows)
        lo, hi = boot(rows, hate_fid, v)
        # directional flips on rows the judge calls NO in full text (counterspeech/quote/non-hate)
        raw_no = [r for r in rows if r["hate"]["raw"] == "NO"]
        no_to_yes = sum(1 for r in raw_no if r["hate"].get(v) == "YES")
        raw_yes = [r for r in rows if r["hate"]["raw"] == "YES"]
        yes_to_no = sum(1 for r in raw_yes if r["hate"].get(v) == "NO")
        unclear = sum(1 for r in rows if r["hate"].get(v) == "UNCLEAR")
        out["hate_fidelity"]["variants"][v] = {
            "n": len(rows), "fidelity_pct": round(100 * fid / len(rows), 1), "ci95": [lo, hi],
            "collapsed_unclear_pct": round(100 * unclear / len(rows), 1),
        }
        out["hate_flip_detail"][v] = {
            "raw_NO_rows": len(raw_no), "NO_to_YES(counterspeech_read_as_hate)": no_to_yes,
            "NO_to_YES_pct": round(100 * no_to_yes / max(len(raw_no), 1), 1),
            "raw_YES_rows": len(raw_yes), "YES_to_NO_pct": round(100 * yes_to_no / max(len(raw_yes), 1), 1),
        }
    out["hate_fidelity"]["mcnemar"] = {
        "icem_vs_importance_only": mcnemar(hate_ref, "icem_context", "importance_only", hate_fid),
        "icem_vs_matched_window": mcnemar(hate_ref, "icem_context", "window_r3_matched", hate_fid),
        "matched_window_vs_importance_only": mcnemar(hate_ref, "window_r3_matched", "importance_only", hate_fid),
    }

    # --- target recovery vs the judge's read of the full raw text ---
    tgt_ref = [r for r in recs if r["target"].get("raw") not in {"none", "", "err"} and toks(r["target"]["raw"])]

    def tgt_ok(r, v):
        return target_match(r["target"].get(v, ""), toks(r["target"]["raw"])) is True

    for v in VARIANTS:
        rows = tgt_ref
        ok = sum(tgt_ok(r, v) for r in rows)
        lo, hi = boot(rows, tgt_ok, v)
        out["target_recovery"][v] = {"n": len(rows), "recovered_vs_raw_pct": round(100 * ok / len(rows), 1), "ci95": [lo, hi]}
    out["target_recovery"]["denominator"] = len(tgt_ref)
    out["target_recovery"]["mcnemar_icem_vs_importance_only"] = mcnemar(tgt_ref, "icem_context", "importance_only", tgt_ok)
    out["target_recovery"]["mcnemar_icem_vs_matched_window"] = mcnemar(tgt_ref, "icem_context", "window_r3_matched", tgt_ok)

    (D / "summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
