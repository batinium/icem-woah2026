#!/usr/bin/env bash
# Build the double-blind WOAH supplement zip. Includes reproduction code, tests,
# and aggregate results/summaries (configs, metrics, judge summaries).
# EXCLUDES: raw datasets / real release text (none live under results/, verified),
# paper source + LaTeX build artifacts, internal scripts, judge checkpoint dumps,
# and environment.md (carries absolute /Users/... paths). After staging it SCANS
# for any /Users/<user> path, the LAN judge IP, or release-text fields and ABORTS
# if found, so nothing identifying or sensitive ships.
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1

STAGE="${1:-/tmp/icem_supplement}"
OUT="${2:-/tmp/icem_supplement.zip}"
rm -rf "$STAGE" "$OUT"; mkdir -p "$STAGE/scripts" "$STAGE/summaries"

# --- code + env + tests ---
cp environment.yml "$STAGE/"
cp docs/paper_woah_2026/supplement_README.md "$STAGE/README.md"
rsync -a --exclude='__pycache__' --exclude='*.pyc' \
  --exclude='create_author_sanity_audit.py' \
  --exclude='summarize_author_sanity_audit.py' \
  --exclude='generate_manual_review.py' \
  --exclude='summarize_manual_review.py' \
  --exclude='overnight_judge_panel.sh' \
  scripts/ "$STAGE/scripts/"
rsync -a --exclude='__pycache__' --exclude='*.pyc' tests/ "$STAGE/tests/"

# --- results: configs, manifests, metrics, judge summaries ---
# Checkpoint files are excluded: run 025 checkpoints store release-text excerpts
# ('tok' field); run 026 checkpoints are large judge-decision dumps whose findings
# are fully captured in the summary markdown files.
rsync -a \
  --exclude='OVERNIGHT_PROGRESS.md' --exclude='PANEL_DONE.marker' \
  --exclude='025_stance_llm_judge/checkpoint.jsonl' \
  --exclude='025_stance_llm_judge/checkpoint_validate.jsonl' \
  --exclude='026_label_recovery_judge/checkpoint.jsonl' \
  --exclude='026_label_recovery_judge/checkpoint_validate.jsonl' \
  --exclude='026_label_recovery_judge/by_model/*/checkpoint.jsonl' \
  docs/paper_woah_2026/results/ "$STAGE/results/"

# --- aggregate summary markdown (curated; no internal working notes or handoffs) ---
# Excluded: claims_audit.md (internal workflow rules), paper_tables.md (draft
# working notes), results.md (internal "Paper use:" annotations throughout;
# per-run README.md files inside results/ already document each run).
for f in sensitivity.md \
         seed_variance_summary.md rationale_overlap_summary.md \
         hatecheck_context_summary.md no_pii_robustness_summary.md \
         model_transfer_summary.md selector_tradeoff_summary.md; do
  [ -f "docs/paper_woah_2026/$f" ] && cp "docs/paper_woah_2026/$f" "$STAGE/summaries/"
done

# --- SAFETY SCAN: abort if anything identifying / sensitive slipped in. Exclude
# this script itself (it contains the search patterns) and match real-text only as
# a JSON data key ("released_text":), not code that references the field name. ---
SELF="$STAGE/scripts/make_supplement.sh"
bad=0
if grep -rlE '/Users/[a-zA-Z0-9_.-]+/' "$STAGE" 2>/dev/null | grep -v "$SELF" | grep .; then echo "!! absolute user path above"; bad=1; fi
if grep -rlE '(192|10)\.[0-9]+\.[0-9]+\.[0-9]+' "$STAGE" 2>/dev/null | grep -v "$SELF" | grep .; then echo "!! private LAN IP above"; bad=1; fi
# real-text risk is DATA dumps, not source that references the field name: scan
# only results/ data files (.json/.jsonl/.csv) for populated real-text keys.
# 'tok' is also a real-text key used by the stance-judge checkpoints (run 025).
if find "$STAGE/results" \( -name '*.jsonl' -o -name '*.json' -o -name '*.csv' \) -print0 2>/dev/null \
     | xargs -0 grep -lE '"(released_text|with_replaced_text|tok)"[[:space:]]*:[[:space:]]*"' 2>/dev/null | grep .; then
  echo "!! real release text in a results data file above"; bad=1; fi
if [ "$bad" -ne 0 ]; then echo "ABORT: scan found sensitive content; zip not built."; exit 2; fi

# --- zip (deterministic-ish; exclude any stray junk) ---
( cd "$STAGE" && zip -qr -X "$OUT" . -x '*.DS_Store' )
echo "OK -> $OUT"
echo "manifest:"; unzip -l "$OUT" | tail -n +2 | awk '{print $4}' | grep -E '/$|\.[a-z]+$' | sed 's#^#  #' | head -40
echo "  ... ($(unzip -l "$OUT" | tail -1 | awk '{print $2}') files, $(du -h "$OUT" | cut -f1))"
