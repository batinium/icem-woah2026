"""Independent LLM label/target recoverability judge for I-CEM releases.

Tests the *harm* claim: does evidence-minimized release corrupt the hate label and
the target a reader recovers? An independent local LLM (LM Studio, LAN-only) is
asked, given ONLY an excerpt, (1) is this hateful/offensive toward a group? and
(2) which group is targeted? We compare the judge's read of each release excerpt to
its read of the full raw text. A release that flips the hate judgment or loses the
target is corrupting label-relevant content.

Population: the 228 HateXplain rows whose full text reads as a non-assertion stance
(negation/quotation/counterspeech) -- exactly where naive minimization is expected
to misread counterspeech/quoted hate as bare assertion. No real text leaves the LAN.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.icem.context_rules import cue_label_for_token
from scripts.icem.tokenizer import tokenize_with_offsets

# LAN-local LM Studio judge; override with ICEM_JUDGE_URL. Default is localhost so
# the supplement is portable — point it at your own OpenAI-compatible server.
URL = os.environ.get("ICEM_JUDGE_URL", "http://localhost:1234/v1/chat/completions")
# Judge model is overridable for multi-judge robustness. Default keeps the
# original headline run (026) reproducible. Additional judges write to a
# per-model subdir (results/026_label_recovery_judge/by_model/<slug>/).
MODEL = os.environ.get("ICEM_JUDGE_MODEL", "qwen/qwen3.5-9b")
# Tool-calling mode forces a structured tool call instead of free text. Required
# for reasoning-tuned judges (gpt-oss, qwen3.6, gemma-4-12b) that otherwise emit
# their answer in `reasoning_content` and leave `content` empty. Set ICEM_JUDGE_TOOLS=1.
TOOLS_MODE = os.environ.get("ICEM_JUDGE_TOOLS", "") not in ("", "0", "false")


def model_slug(m):
    return m.replace("/", "__").replace(":", "_")

HATE_SYS = (
    "You judge whether a short text excerpt is hateful or offensive toward a group of people. "
    "The excerpt may be fragmentary (words joined by gaps or [MASKED] placeholders). "
    "Answer with exactly one word:\n"
    "YES = the excerpt expresses or endorses hate/offence toward a group.\n"
    "NO = the excerpt does not (including when it denies, quotes-without-endorsing, or opposes such content).\n"
    "UNCLEAR = too fragmentary to tell.\n"
    "Reply with ONLY YES, NO, or UNCLEAR."
)
TARGET_SYS = (
    "You identify which group of people a short text excerpt targets or is about. "
    "The excerpt may be fragmentary. Reply with the group in 1-3 lowercase words "
    "(e.g. 'muslims', 'gay people', 'women', 'black people', 'immigrants', 'jews'). "
    "If no group is identifiable, reply NONE. Reply with ONLY the group or NONE."
)


def _post(payload: dict, retries: int = 4):
    body = json.dumps(payload).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
            return json.loads(urllib.request.urlopen(req, timeout=180).read())
        except Exception as e:  # noqa: BLE001
            if attempt == retries - 1:
                return {"_err": str(e)}
            time.sleep(1.5 * (attempt + 1))
    return {"_err": "retries exhausted"}


def call(sys_prompt: str, text: str, max_tokens: int = 8) -> str:
    r = _post({
        "model": MODEL,
        "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": text}],
        "temperature": 0, "max_tokens": max_tokens,
    })
    if "_err" in r:
        return f"ERR:{r['_err']}"
    return r["choices"][0]["message"]["content"].strip()


def call_tool(sys_prompt: str, text: str, tool: dict, max_tokens: int = 768) -> dict:
    """Force a structured tool call; return its parsed arguments dict (or {})."""
    # NB: explicit forced tool_choice ({"type":"function",...}) returns HTTP 400 on
    # harmony models (gpt-oss); "auto" + a tool-only context yields a clean tool call
    # across the whole panel. Content fallback covers any prose answer.
    r = _post({
        "model": MODEL,
        "messages": [{"role": "system", "content": sys_prompt + "\nAlways answer by calling the provided tool."},
                     {"role": "user", "content": text}],
        "temperature": 0, "max_tokens": max_tokens,
        "tools": [tool],
        "tool_choice": "auto",
    })
    if "_err" in r:
        return {"_err": r["_err"]}
    msg = r["choices"][0]["message"]
    tcs = msg.get("tool_calls") or []
    if tcs:
        try:
            return json.loads(tcs[0]["function"]["arguments"])
        except Exception:  # noqa: BLE001
            return {"_raw": tcs[0]["function"].get("arguments", "")}
    # model answered in prose despite forced tool_choice -> fall back to content
    return {"_content": (msg.get("content") or "").strip()}


HATE_TOOL = {"type": "function", "function": {
    "name": "record_hate_judgment",
    "description": "Record whether the excerpt is hateful/offensive toward a group.",
    "parameters": {"type": "object", "properties": {
        "label": {"type": "string", "enum": ["YES", "NO", "UNCLEAR"],
                  "description": "YES=expresses/endorses hate toward a group; NO=does not "
                                 "(incl. denial, quote-without-endorsing, opposition); UNCLEAR=too fragmentary."}},
        "required": ["label"]}}}
TARGET_TOOL = {"type": "function", "function": {
    "name": "record_target_group",
    "description": "Record which group of people the excerpt targets or is about.",
    "parameters": {"type": "object", "properties": {
        "group": {"type": "string", "description": "The targeted group in 1-3 lowercase words "
                  "(e.g. 'muslims', 'gay people', 'women'), or 'none' if no group is identifiable."}},
        "required": ["group"]}}}


def hate(text: str) -> str:
    if TOOLS_MODE:
        args = call_tool(HATE_SYS, text, HATE_TOOL)
        out = str(args.get("label") or args.get("_content") or "").upper()
    else:
        out = call(HATE_SYS, text).upper()
    for lab in ("UNCLEAR", "YES", "NO"):
        if lab in out:
            return lab
    return "UNCLEAR"


def target(text: str) -> str:
    if TOOLS_MODE:
        args = call_tool(TARGET_SYS, text, TARGET_TOOL)
        return str(args.get("group") or args.get("_content") or "").lower().strip().strip(".")
    return call(TARGET_SYS, text, max_tokens=12).lower().strip().strip(".")


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
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    base = ROOT / "docs/paper_woah_2026/results/026_label_recovery_judge"
    # Plain-completion qwen3.5-9b reproduces the committed run 026 in the base dir.
    # Every tool-call judge (incl. tool-call qwen3.5-9b) is a separate by_model run,
    # so the committed headline is never clobbered.
    if MODEL == "qwen/qwen3.5-9b" and not TOOLS_MODE:
        outdir = base
    else:
        outdir = base / "by_model" / model_slug(MODEL)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"judge model: {MODEL}\noutdir: {outdir}", flush=True)
    ckpt = outdir / ("checkpoint_validate.jsonl" if limit else "checkpoint.jsonl")

    r015 = load_run(ROOT / "data/outputs/015_hatexplain_dehatebert_5000_no_pii/release_rows.jsonl")
    r022 = load_run(ROOT / "data/outputs/022_hatexplain_nopii_radius3_5000/022_hatexplain_nopii_radius3_5000/release_rows.jsonl")

    cands = sorted(rid for rid, r in r015.items()
                   if stance_relevant(r["row"]["text"], r["row"].get("target_groups", [])))
    # Deterministic subsample (seed 17) for tractable multi-judge runs. SAME rows
    # across every judge -> cross-judge metrics stay comparable. ICEM_JUDGE_SAMPLE=0
    # (default) keeps the full candidate set (the committed run 026 uses full).
    sample = int(os.environ.get("ICEM_JUDGE_SAMPLE", "0"))
    if sample and len(cands) > sample:
        import random as _r
        cands = sorted(_r.Random(17).sample(cands, sample))
    if limit:
        cands = cands[:limit]
    print(f"candidates: {len(cands)}", flush=True)

    done = set()
    if ckpt.exists():
        for line in ckpt.open():
            try:
                done.add(json.loads(line)["row_id"])
            except Exception:  # noqa: BLE001
                pass
    print(f"already done: {len(done)}", flush=True)

    def work(rid):
        r = r015[rid]
        raw = r["row"]["text"]
        v = r["variants"]
        excerpts = {
            "raw": raw,
            "importance_only": v["importance_only"]["released_text"],
            "importance_window_r2": v["importance_window"]["released_text"],
            "icem_context": v["icem_context"]["released_text"],
        }
        if rid in r022:
            excerpts["window_r3_matched"] = r022[rid]["variants"]["importance_window"]["released_text"]
        return {
            "row_id": rid,
            "binary_label": r["row"]["binary_label"],
            "gold_targets": r["row"].get("target_groups", []),
            "hate": {k: hate(t) for k, t in excerpts.items()},
            "target": {k: target(t) for k, t in excerpts.items()},
        }

    todo = [rid for rid in cands if rid not in done]
    t0 = time.time(); n = 0
    with ckpt.open("a") as fh, ThreadPoolExecutor(max_workers=workers) as ex:
        for rec in ex.map(work, todo):
            fh.write(json.dumps(rec) + "\n"); fh.flush(); n += 1
            if n % 20 == 0:
                rate = n / (time.time() - t0)
                print(f"  {n}/{len(todo)}  {rate:.2f} rows/s  eta {(len(todo)-n)/max(rate,1e-9):.0f}s", flush=True)
    print("done judging; run analyze_label_recovery.py", flush=True)


if __name__ == "__main__":
    main()
