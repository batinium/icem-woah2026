"""Independent LLM stance-sufficiency judge for I-CEM releases.

Lexicon-free external grounding for the Stance proxy. For each stance-relevant
HateXplain row (no-PII run 015), an independent local LLM (LM Studio) labels the
author's stance toward the harmful content in (a) the full raw text -> reference,
and (b) each release excerpt. A release "preserves stance" if its excerpt still
reads as the same non-assertion stance the judge reads from the full text.

This does NOT use I-CEM's cue lexicon, breaking the circularity in the proxy.
All inference is on a LAN-local model; no real text leaves the local network.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.icem.context_rules import cue_label_for_token
from scripts.icem.tokenizer import tokenize_with_offsets

# LAN-local LM Studio judge; override with ICEM_JUDGE_URL (default localhost for
# a portable supplement).
URL = os.environ.get("ICEM_JUDGE_URL", "http://localhost:1234/v1/chat/completions")
MODEL = "qwen/qwen3.5-9b"

SYS = (
    "You label the AUTHOR'S STANCE toward harmful or hateful content in a short text "
    "excerpt. The excerpt may be fragmentary (words joined by gaps). Pick exactly one:\n"
    "ASSERT = the author states the harmful claim as their own view.\n"
    "NEGATE = the harmful claim is explicitly denied, refused, or said to be false.\n"
    "QUOTE  = the harmful claim is explicitly quoted or attributed to another person, not endorsed.\n"
    "OPPOSE = the author explicitly condemns, rejects, or argues against the harmful claim.\n"
    "UNCLEAR = too fragmentary or missing context to tell the stance.\n"
    "Use NEGATE/QUOTE/OPPOSE only when the excerpt itself shows that signal. "
    "If the excerpt is just a bare harmful phrase with no negation/attribution/opposition visible, "
    "label ASSERT (or UNCLEAR if it is only disconnected words).\n"
    "Reply with ONLY the single label word."
)

LABELS = ("ASSERT", "NEGATE", "QUOTE", "OPPOSE", "UNCLEAR")
NONASSERT = {"NEGATE", "QUOTE", "OPPOSE"}


def judge(text: str, retries: int = 4) -> str:
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": text}],
        "temperature": 0,
        "max_tokens": 6,
    }).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
            r = json.loads(urllib.request.urlopen(req, timeout=120).read())
            out = r["choices"][0]["message"]["content"].upper()
            for lab in LABELS:
                if lab in out:
                    return lab
            return "UNCLEAR"
        except Exception as e:  # noqa: BLE001
            if attempt == retries - 1:
                return f"ERR:{e}"
            time.sleep(1.5 * (attempt + 1))
    return "ERR"


def cues(text, tgroups):
    labs = set()
    for tok in tokenize_with_offsets(text):
        l = cue_label_for_token(tok.text, tgroups)
        if l:
            labs.add(l)
    return labs


def stance_relevant(text, tgroups):
    c = cues(text, tgroups)
    return bool(c & {"NEGATION_CUE", "QUOTE_CUE", "COUNTERSPEECH_CUE"}) and "HARM_CUE" in c


def load_run(path):
    rows = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            rows[r["row"]["row_id"]] = r
    return rows


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    outdir = ROOT / "docs/paper_woah_2026/results/025_stance_llm_judge"
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt = outdir / ("checkpoint_validate.jsonl" if limit else "checkpoint.jsonl")

    r015 = load_run(ROOT / "data/outputs/015_hatexplain_dehatebert_5000_no_pii/release_rows.jsonl")
    r022 = load_run(ROOT / "data/outputs/022_hatexplain_nopii_radius3_5000/022_hatexplain_nopii_radius3_5000/release_rows.jsonl")

    candidates = []
    for rid, r in r015.items():
        if stance_relevant(r["row"]["text"], r["row"].get("target_groups", [])):
            candidates.append(rid)
    candidates.sort()  # deterministic
    if limit:
        candidates = candidates[:limit]
    print(f"stance-relevant candidates: {len(candidates)}", flush=True)

    done = {}
    if ckpt.exists():
        for line in ckpt.open():
            try:
                rec = json.loads(line)
                done[rec["row_id"]] = rec
            except Exception:  # noqa: BLE001
                pass
    print(f"resuming, already done: {len(done)}", flush=True)

    def work(rid):
        if rid in done:
            return done[rid]
        r = r015[rid]
        raw = r["row"]["text"]
        v = r["variants"]
        excerpts = {
            "raw": raw,
            "importance_only": v["importance_only"]["released_text"],
            "importance_window_r2": v["importance_window"]["released_text"],
            "icem_context": v["icem_context"]["released_text"],
        }
        # matched-budget window r=3 from run 022
        if rid in r022:
            excerpts["window_r3_matched"] = r022[rid]["variants"]["importance_window"]["released_text"]
        labels = {k: judge(t) for k, t in excerpts.items()}
        rec = {"row_id": rid, "binary_label": r["row"]["binary_label"], "labels": labels,
               "tok": {k: v[k]["released_text"] for k in ("importance_only", "importance_window", "icem_context")}}
        return rec

    t0 = time.time()
    n = 0
    with ckpt.open("a") as fh, ThreadPoolExecutor(max_workers=workers) as ex:
        for rec in ex.map(work, candidates):
            if rec["row_id"] not in done:
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
            n += 1
            if n % 25 == 0:
                rate = n / (time.time() - t0)
                print(f"  {n}/{len(candidates)}  {rate:.2f} rows/s  eta {(len(candidates)-n)/max(rate,1e-9):.0f}s", flush=True)

    # aggregate
    recs = []
    for line in ckpt.open():
        try:
            recs.append(json.loads(line))
        except Exception:  # noqa: BLE001
            pass
    summarize(recs, outdir, limit)


def summarize(recs, outdir, limit):
    variants = ["importance_only", "importance_window_r2", "icem_context", "window_r3_matched"]
    # reference = judge on raw; keep rows where raw is a non-assertion stance (stance matters)
    ref_rows = [r for r in recs if r["labels"].get("raw") in NONASSERT]
    err = sum(1 for r in recs if any(str(v).startswith("ERR") for v in r["labels"].values()))
    out = {
        "model": MODEL,
        "n_candidates_judged": len(recs),
        "n_with_judge_errors": err,
        "denominator_raw_nonassertion": len(ref_rows),
        "raw_label_distribution": _dist([r["labels"].get("raw") for r in recs]),
        "metrics": {},
    }
    for var in variants:
        rows = [r for r in ref_rows if var in r["labels"] and not str(r["labels"][var]).startswith("ERR")]
        if not rows:
            continue
        preserved = sum(1 for r in rows if r["labels"][var] in NONASSERT)          # kept any non-assertion stance
        exact = sum(1 for r in rows if r["labels"][var] == r["labels"]["raw"])      # same stance category
        collapsed_assert = sum(1 for r in rows if r["labels"][var] == "ASSERT")
        collapsed_unclear = sum(1 for r in rows if r["labels"][var] == "UNCLEAR")
        out["metrics"][var] = {
            "n": len(rows),
            "stance_preserved_pct": round(100 * preserved / len(rows), 1),
            "exact_stance_match_pct": round(100 * exact / len(rows), 1),
            "collapsed_to_assert_pct": round(100 * collapsed_assert / len(rows), 1),
            "collapsed_to_unclear_pct": round(100 * collapsed_unclear / len(rows), 1),
        }
    name = "summary_validate.json" if limit else "summary.json"
    (outdir / name).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)


def _dist(labels):
    d = {}
    for l in labels:
        d[l] = d.get(l, 0) + 1
    return d


if __name__ == "__main__":
    main()
