#!/usr/bin/env python3
"""Create a compact, sanitized author manual-audit CSV for I-CEM outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import re
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.icem.context_rules import HARM_CUES, TARGET_CUES, cue_label_for_token
from scripts.icem.tokenizer import tokenize_with_offsets


AUDIT_FIELDS = (
    "audit_id",
    "row_id",
    "dataset",
    "split",
    "label",
    "binary_label",
    "target_groups",
    "selection_reason",
    "source_sanitized",
    "importance_only",
    "importance_window",
    "icem_context",
    "importance_only_context_sufficient",
    "importance_window_context_sufficient",
    "icem_context_sufficient",
    "best_context_variant",
    "icem_not_more_misleading_than_window",
    "obvious_privacy_risk_remaining",
    "review_notes",
)

ANSWER_GUIDE = {
    "importance_only_context_sufficient": "yes/no/uncertain",
    "importance_window_context_sufficient": "yes/no/uncertain",
    "icem_context_sufficient": "yes/no/uncertain",
    "best_context_variant": "importance_only/importance_window/icem_context/tie/none/uncertain",
    "icem_not_more_misleading_than_window": "yes/no/uncertain",
    "obvious_privacy_risk_remaining": "yes/no/uncertain",
}

PROTECTED_GROUP_TERMS = TARGET_CUES | {
    "african",
    "africans",
    "american",
    "americans",
    "catholic",
    "catholics",
    "disabled",
    "disability",
    "female",
    "females",
    "girl",
    "girls",
    "gays",
    "asians",
    "hispanic",
    "hispanics",
    "immigration",
    "indian",
    "indians",
    "islam",
    "jewess",
    "minority",
    "minorities",
    "sikh",
    "sikhs",
}

HARM_PLACEHOLDER_TERMS = HARM_CUES | {
    "ass",
    "bastard",
    "bastards",
    "bitch",
    "bitches",
    "burn",
    "burnin",
    "bullshit",
    "crackhead",
    "crackheads",
    "damn",
    "devil",
    "devils",
    "dick",
    "dicks",
    "dumb",
    "dumbass",
    "fucking",
    "gook",
    "gooks",
    "homophobe",
    "hoe",
    "hoes",
    "hooker",
    "hookers",
    "idiot",
    "idiots",
    "jihadi",
    "jihadis",
    "monkey",
    "monkeys",
    "moron",
    "morons",
    "mudshark",
    "mudsharks",
    "paki",
    "pakis",
    "pervert",
    "perverts",
    "pics",
    "rape",
    "raped",
    "rapists",
    "scum",
    "shitskin",
    "shitskins",
    "shit",
    "sick",
    "stupid",
    "supremacist",
    "supremacists",
    "trash",
    "twat",
    "violence",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-rows", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument(
        "--seed-judgments",
        dest="seed_judgments",
        action="store_true",
        help="Seed review columns with deterministic initial judgments.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bundles = read_bundles(args.release_rows)
    selected = select_rows(bundles, sample_size=args.sample_size, seed=args.seed)
    rows = [
        audit_row(index + 1, bundle, reason, seed_judgments=args.seed_judgments)
        for index, (bundle, reason) in enumerate(selected)
    ]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    args.output_md.write_text(instructions(args.output_csv, rows), encoding="utf-8")
    return 0


def read_bundles(path: Path) -> list[dict[str, Any]]:
    bundles: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            variants = payload.get("variants", {})
            if {"pii_quasi_mask", "importance_only", "importance_window", "icem_context"} <= set(variants):
                bundles.append(payload)
    return bundles


def select_rows(
    bundles: list[dict[str, Any]],
    *,
    sample_size: int,
    seed: int,
) -> list[tuple[dict[str, Any], str]]:
    rng = random.Random(seed)
    by_label: dict[str, list[tuple[dict[str, Any], str]]] = {}
    for bundle in bundles:
        label = str(bundle["row"].get("label", "unknown"))
        by_label.setdefault(label, []).append((bundle, selection_reason(bundle)))

    labels = sorted(by_label)
    per_label = max(1, sample_size // max(1, len(labels)))
    remainder = sample_size - per_label * len(labels)
    selected: list[tuple[dict[str, Any], str]] = []
    reason_order = (
        "icem_adds_context",
        "importance_only_sparse",
        "window_fragment",
        "possible_failure",
        "label_balance",
    )

    for offset, label in enumerate(labels):
        target = per_label + int(offset < remainder)
        candidates = by_label[label][:]
        rng.shuffle(candidates)
        picked: list[tuple[dict[str, Any], str]] = []
        buckets = {
            reason: [(bundle, bundle_reason) for bundle, bundle_reason in candidates if bundle_reason == reason]
            for reason in reason_order
        }
        while len(picked) < target and any(buckets.values()):
            made_progress = False
            for reason in reason_order:
                while buckets[reason] and buckets[reason][0][0] in [item[0] for item in picked]:
                    buckets[reason].pop(0)
                if buckets[reason] and len(picked) < target:
                    picked.append(buckets[reason].pop(0))
                    made_progress = True
            if not made_progress:
                break
        for bundle, bundle_reason in candidates:
            if len(picked) >= target:
                break
            if bundle not in [item[0] for item in picked]:
                picked.append((bundle, bundle_reason))
        selected.extend(picked)

    rng.shuffle(selected)
    return selected[:sample_size]


def selection_reason(bundle: dict[str, Any]) -> str:
    variants = bundle["variants"]
    importance = variants["importance_only"]["released_text"]
    window = variants["importance_window"]["released_text"]
    icem = variants["icem_context"]["released_text"]
    if token_count(icem) >= token_count(window) + 4:
        return "icem_adds_context"
    if token_count(importance) <= 2:
        return "importance_only_sparse"
    if looks_fragmentary(window):
        return "window_fragment"
    if token_count(icem) <= 6 or normalized(icem) == normalized(window):
        return "possible_failure"
    return "label_balance"


def audit_row(
    audit_id: int,
    bundle: dict[str, Any],
    reason: str,
    *,
    seed_judgments: bool = False,
) -> dict[str, str]:
    row = bundle["row"]
    variants = bundle["variants"]
    target_groups = tuple(str(value) for value in row.get("target_groups", ()))
    values = {
        "audit_id": str(audit_id),
        "row_id": str(row.get("row_id", "")),
        "dataset": str(row.get("dataset", "")),
        "split": str(row.get("split", "")),
        "label": str(row.get("label", "")),
        "binary_label": "" if row.get("binary_label") is None else str(row.get("binary_label")),
        "target_groups": f"{len(target_groups)} group(s)",
        "selection_reason": reason,
        "source_sanitized": sanitize(variants["pii_quasi_mask"]["released_text"], target_groups),
        "importance_only": sanitize(variants["importance_only"]["released_text"], target_groups),
        "importance_window": sanitize(variants["importance_window"]["released_text"], target_groups),
        "icem_context": sanitize(variants["icem_context"]["released_text"], target_groups),
        "review_notes": "",
    }
    if seed_judgments:
        values.update(seed_judgment_values(values))
    for field in AUDIT_FIELDS:
        values.setdefault(field, "")
    return values


CONTEXT_TERMS = {
    "against",
    "called",
    "claim",
    "claimed",
    "condemn",
    "condemned",
    "criticized",
    "fake",
    "not",
    "opposed",
    "quote",
    "quoted",
    "reject",
    "rejected",
    "said",
    "says",
    "shared",
    "stop",
    "support",
    "supporting",
    "victim",
    "wrote",
}

PRIVACY_RISK_PATTERN = re.compile(
    r"\b\d{3}[-.) ]?\d{3}[- ]?\d{4}\b|"
    r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b|"
    r"https?://\S+|www\.\S+|"
    r"@\w+"
)


def seed_judgment_values(row: dict[str, str]) -> dict[str, str]:
    variants = {
        "importance_only": row["importance_only"],
        "importance_window": row["importance_window"],
        "icem_context": row["icem_context"],
    }
    sufficiency = {
        name: context_sufficient(text, row["label"]) for name, text in variants.items()
    }
    scores = {name: context_score(text, row["label"]) for name, text in variants.items()}
    best = best_variant(scores, variants)
    window_score = scores["importance_window"]
    icem_score = scores["icem_context"]
    icem_sufficient = sufficiency["icem_context"]
    window_sufficient = sufficiency["importance_window"]
    if icem_score >= window_score and icem_sufficient != "no":
        not_more_misleading = "yes"
    elif icem_sufficient == "uncertain" or window_sufficient == "uncertain":
        not_more_misleading = "uncertain"
    else:
        not_more_misleading = "no"
    return {
        "importance_only_context_sufficient": sufficiency["importance_only"],
        "importance_window_context_sufficient": sufficiency["importance_window"],
        "icem_context_sufficient": icem_sufficient,
        "best_context_variant": best,
        "icem_not_more_misleading_than_window": not_more_misleading,
        "obvious_privacy_risk_remaining": privacy_risk_remaining(row),
        "review_notes": "",
    }


def context_sufficient(text: str, label: str) -> str:
    if text == "[NO_TASK_EVIDENCE_RETAINED]":
        return "no"
    count = token_count(text)
    if count <= 3:
        return "no"
    has_harm = "[HARM]" in text
    has_group = "[GROUP]" in text
    has_context = contains_context_term(text)
    label_normal = normalized(label) == "normal"
    if label_normal:
        if count >= 10 and has_context:
            return "yes"
        if count >= 6:
            return "uncertain"
        return "no"
    if has_harm and (has_group or has_context or count >= 8):
        return "yes"
    if has_group and has_context and count >= 6:
        return "yes"
    if count >= 10 or (has_harm and count >= 5):
        return "uncertain"
    return "no"


def context_score(text: str, label: str) -> int:
    if text == "[NO_TASK_EVIDENCE_RETAINED]":
        return -20
    score = min(token_count(text), 18)
    if "[HARM]" in text:
        score += 4
    if "[GROUP]" in text:
        score += 4
    if contains_context_term(text):
        score += 4
    if normalized(label) == "normal" and "[HARM]" not in text:
        score += 2
    if looks_fragmentary(text):
        score -= 2
    return score


def best_variant(scores: dict[str, int], variants: dict[str, str]) -> str:
    if all(score <= 0 for score in scores.values()):
        return "none"
    best_name = max(scores, key=lambda key: (scores[key], key == "icem_context"))
    best_score = scores[best_name]
    tied = [
        name
        for name, score in scores.items()
        if score == best_score and normalized(variants[name]) == normalized(variants[best_name])
    ]
    if len(tied) > 1:
        return "tie"
    return best_name


def contains_context_term(text: str) -> bool:
    return bool(CONTEXT_TERMS & set(re.findall(r"[a-z']+", text.lower())))


def privacy_risk_remaining(row: dict[str, str]) -> str:
    visible_text = " ".join(
        row[field]
        for field in (
            "source_sanitized",
            "importance_only",
            "importance_window",
            "icem_context",
        )
    )
    scrubbed = re.sub(
        r"\[(?:EMAIL|URL|HANDLE|PERSON|PHONE|SCHOOL|WORKPLACE|EVENT)\]",
        "",
        visible_text,
    )
    return "yes" if PRIVACY_RISK_PATTERN.search(scrubbed) else "no"


def sanitize(text: str, target_groups: tuple[str, ...]) -> str:
    if text == "[NO_TASK_EVIDENCE_RETAINED]":
        return text
    tokens = tokenize_with_offsets(text)
    pieces: list[str] = []
    cursor = 0
    for token in tokens:
        pieces.append(text[cursor : token.start])
        pieces.append(replacement_for_token(token.text, target_groups))
        cursor = token.end
    pieces.append(text[cursor:])
    sanitized = "".join(pieces)
    sanitized = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", "[EMAIL]", sanitized)
    sanitized = re.sub(r"https?://\S+|www\.\S+", "[URL]", sanitized)
    sanitized = re.sub(r"@\w+", "[HANDLE]", sanitized)
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    return sanitized


def replacement_for_token(text: str, target_groups: tuple[str, ...]) -> str:
    lowered = normalized(text)
    if not lowered:
        return text
    label = cue_label_for_token(lowered, target_groups)
    if label == "TARGET_CUE" or lowered in PROTECTED_GROUP_TERMS:
        return "[GROUP]"
    if label == "HARM_CUE" or lowered in HARM_PLACEHOLDER_TERMS:
        return "[HARM]"
    if lowered.startswith(("fuck", "nigg", "fag", "kik", "shit", "rape", "rapist")):
        return "[HARM]"
    if lowered.endswith(("slur", "slurs")):
        return "[HARM]"
    return text


def normalized(text: str) -> str:
    return re.sub(r"[^a-z0-9']+", "", text.lower())


def token_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b|\[[A-Z_]+\]", text))


def looks_fragmentary(text: str) -> bool:
    count = token_count(text)
    if count <= 4:
        return True
    stripped = text.strip()
    return stripped.startswith("...") or stripped.endswith("...") or stripped.count("...") >= 2


def instructions(csv_path: Path, rows: list[dict[str, str]]) -> str:
    reason_counts: dict[str, int] = {}
    label_counts: dict[str, int] = {}
    for row in rows:
        reason_counts[row["selection_reason"]] = reason_counts.get(row["selection_reason"], 0) + 1
        label_counts[row["label"]] = label_counts.get(row["label"], 0) + 1

    lines = [
        "# Author Manual Audit",
        "",
        f"CSV: `{csv_path}`",
        "",
        "This is a compact author manual audit, not a formal annotation study.",
        "The rows are sanitized: group and harm cues are placeholders, and the",
        "source column uses the PII/quasi-PII masked release rather than raw text.",
        "Review and correct any judgment that looks wrong during author audit.",
        "",
        "Review only these columns:",
        "",
    ]
    for field, guide in ANSWER_GUIDE.items():
        lines.append(f"- `{field}`: `{guide}`")
    lines.extend(
        [
            "- `review_notes`: optional short note for failures or uncertainty",
            "",
            "Recommended rule of thumb:",
            "",
            "- Mark context sufficient if the reduced text preserves enough target,",
            "  harm, negation/quotation/counterspeech, and stance information to",
            "  understand why the original label could be assigned.",
            "- Mark misleading if the reduction changes the apparent stance, for",
            "  example by dropping `not`, `quoted`, `criticized`, or similar context.",
            "- Mark privacy risk only for visible identifiers or unusually specific",
            "  quasi-identifiers that remain after masking.",
            "",
            "Sample composition:",
            "",
            f"- Rows: {len(rows)}",
            f"- Labels: {label_counts}",
            f"- Selection reasons: {reason_counts}",
            "",
            "After filling the CSV, run:",
            "",
            "```bash",
            "micromamba run -n icem-research python scripts/summarize_author_sanity_audit.py \\",
            f"  --input-csv {csv_path} \\",
            "  --output-md docs/paper_woah_2026/author_sanity_audit_summary.md \\",
            "  --output-json docs/paper_woah_2026/author_sanity_audit_summary.json",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
