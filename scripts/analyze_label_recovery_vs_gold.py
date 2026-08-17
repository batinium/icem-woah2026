#!/usr/bin/env python3
"""Validate the LLM label/target-recovery judge against the datasets' own gold
labels, requiring no new human annotation.

Two questions:
  1. Is the judge a usable instrument? -> agreement of its full-text hate read
     with the gold binary_label.
  2. Does the I-CEM > token-matched-window target-recovery ordering depend on the
     judge, or does it survive when scored against the gold target groups?

Input: results/026_label_recovery_judge/checkpoint.jsonl, whose records carry
both the gold fields (binary_label, gold_targets) and the judge's per-variant
reads (hate, target). Reference run 026 (qwen3.5-9b, seed 17, 963 rows).

Usage:
  python scripts/analyze_label_recovery_vs_gold.py \
      docs/paper_woah_2026/results/026_label_recovery_judge/checkpoint.jsonl
"""
import json
import sys
from math import comb

# HateXplain uses a small controlled target vocabulary (african, homosexual,
# islam, ...); the lexicon-free judge answers in natural language (black people,
# gay people, muslims, ...). This map aligns the two vocabularies so a correct
# but differently-worded judge target counts as a match against gold.
SYN = {
    "african": {"black", "blacks", "african", "africans"},
    "caucasian": {"white", "whites", "caucasian"},
    "islam": {"muslim", "muslims", "moslem", "moslems", "muzzies", "islam", "islamic"},
    "jewish": {"jew", "jews", "jewish"},
    "homosexual": {"gay", "gays", "queer", "queers", "lesbian", "homosexual", "lgbt", "fag", "faggot"},
    "hispanic": {"hispanic", "latino", "latina", "mexican", "mexicans", "illegal", "immigrant", "immigrants"},
    "refugee": {"refugee", "refugees", "immigrant", "immigrants", "migrant", "migrants"},
    "women": {"women", "woman", "female", "females", "girl", "girls"},
    "men": {"men", "man", "male", "males"},
    "arab": {"arab", "arabs"},
    "asian": {"asian", "asians", "chinese", "japanese"},
    "indian": {"indian", "indians", "paki", "pakis"},
    "disability": {"disabled", "disability", "retard", "autistic"},
    "christian": {"christian", "christians"},
    "hindu": {"hindu", "hindus"},
    "indigenous": {"indigenous", "native"},
    "nonreligious": {"atheist", "atheists"},
}
VARIANTS = ["importance_only", "importance_window_r2", "window_r3_matched", "icem_context"]


def mcnemar_exact(n10, n01):
    n = n10 + n01
    if n == 0:
        return 1.0
    k = min(n10, n01)
    p = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * p)


def jtokens(s):
    return set(str(s).lower().replace(",", " ").replace("/", " ").split())


def target_overlap(judge_t, gold):
    gold = [g for g in (gold or []) if g and g != "none"]
    if not gold:
        return None  # no gold target -> excluded from denominator
    t = jtokens(judge_t)
    if not t or t == {"none"}:
        return False
    for g in gold:
        g = str(g).lower()
        if t & SYN.get(g, {g}) or g in t:
            return True
    return False


def hate_correct(read, gold_binary):
    if read == "UNCLEAR":
        return None
    return (1 if read == "YES" else 0) == gold_binary


def paired_mcnemar(a, b):
    """a, b: lists of bool correctness. Returns (b_only, a_only, p)."""
    n10 = sum(1 for x, y in zip(a, b) if y and not x)  # b correct, a wrong
    n01 = sum(1 for x, y in zip(a, b) if x and not y)  # a correct, b wrong
    return n10, n01, mcnemar_exact(n10, n01)


def main(path):
    rows = [json.loads(l) for l in open(path)]
    print(f"rows: {len(rows)}\n")

    # 1. Judge instrument validation: full-text hate read vs gold binary_label
    tp = fp = tn = fn = 0
    for r in rows:
        c = hate_correct(r["hate"]["raw"], r["binary_label"])
        if c is None:
            continue
        pred = 1 if r["hate"]["raw"] == "YES" else 0
        g = r["binary_label"]
        tp += pred == 1 and g == 1
        fp += pred == 1 and g == 0
        tn += pred == 0 and g == 0
        fn += pred == 0 and g == 1
    n = tp + fp + tn + fn
    acc = (tp + tn) / n
    rec = tp / (tp + fn)
    prec = tp / (tp + fp)
    print("=== Judge full-text HATE read vs GOLD binary_label ===")
    print(f"agreement={acc*100:.1f}%  recall={rec*100:.1f}%  precision={prec*100:.1f}%  (n={n})\n")

    # 2. Target recovery vs gold (judge-independent reference)
    print("=== Target recovery vs GOLD target groups (synonym-mapped) ===")
    cols = {v: [] for v in VARIANTS}
    judge_raw = []
    for r in rows:
        g = r.get("gold_targets")
        if target_overlap(r["target"]["raw"], g) is None:
            continue
        judge_raw.append(bool(target_overlap(r["target"]["raw"], g)))
        for v in VARIANTS:
            cols[v].append(bool(target_overlap(r["target"][v], g)))
    nt = len(cols["icem_context"])
    for v in VARIANTS:
        print(f"  {v:24s}: {sum(cols[v])/nt*100:.1f}%  (n={nt})")
    print(f"  [judge full-text target vs gold: {sum(judge_raw)/len(judge_raw)*100:.1f}%]")
    b, a, p = paired_mcnemar(cols["window_r3_matched"], cols["icem_context"])
    print(f"  McNemar I-CEM vs matched window: icem-only={b} matched-only={a} p={p:.4f}")
    b, a, p = paired_mcnemar(cols["importance_only"], cols["icem_context"])
    print(f"  McNemar I-CEM vs importance-only: p={p:.2e}\n")

    # 3. Hate fidelity vs gold (paired, drop rows where any variant is UNCLEAR)
    print("=== Hate accuracy vs GOLD binary_label (paired) ===")
    hcols = {v: [] for v in VARIANTS}
    for r in rows:
        vals = {v: hate_correct(r["hate"][v], r["binary_label"]) for v in VARIANTS}
        if any(x is None for x in vals.values()):
            continue
        for v in VARIANTS:
            hcols[v].append(vals[v])
    nh = len(hcols["icem_context"])
    for v in VARIANTS:
        print(f"  {v:24s}: {sum(hcols[v])/nh*100:.1f}%  (n={nh})")
    b, a, p = paired_mcnemar(hcols["window_r3_matched"], hcols["icem_context"])
    print(f"  McNemar I-CEM vs matched window: icem-only={b} matched-only={a} p={p:.4f}")
    b, a, p = paired_mcnemar(hcols["importance_only"], hcols["icem_context"])
    print(f"  McNemar I-CEM vs importance-only: p={p:.2e}\n")

    # 4. Caveat: CS->Hate vs gold non-hate is NOT a clean stance reference,
    # because HateXplain gold labels do not encode stance.
    print("=== CS->Hate vs GOLD non-hate rows (caveat: gold ignores stance) ===")
    gold_no = [r for r in rows if r["binary_label"] == 0]
    print(f"  gold non-hate rows: {len(gold_no)}")
    for v in VARIANTS:
        flip = sum(1 for r in gold_no if r["hate"][v] == "YES")
        print(f"  {v:24s}: {flip/len(gold_no)*100:.1f}% flipped to hate")


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else \
        "docs/paper_woah_2026/results/026_label_recovery_judge/checkpoint.jsonl"
    main(p)
