#!/usr/bin/env bash
# OVERNIGHT autonomous multi-judge robustness run for run 026.
# Runs the tool-calling judge panel sequentially (one model in VRAM at a time),
# analyzes each, then aggregates. Resumable: per-model checkpoints mean a
# re-launch skips finished rows. Progress is appended to a markdown log so a
# fresh context can resume. LM Studio must have per-model "Max instances = 1"
# (verified by safety probe before launch) so WORKERS>1 does not clone instances.
#
# Cleanup note: LM Studio REST unload is a no-op from a remote client; models are
# reclaimed by JIT idle-TTL once requests stop (i.e. when this script ends).
set -uo pipefail   # NOT -e: one bad judge must not abort the panel
cd "$(dirname "$0")/.." || exit 1

RUN="micromamba run -n icem-research"
RESULTS="docs/paper_woah_2026/results/026_label_recovery_judge"
BASE="$RESULTS/by_model"
PROG="$RESULTS/OVERNIGHT_PROGRESS.md"
export ICEM_JUDGE_TOOLS=1 KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1
# LM Studio serializes concurrent requests on a single instance (Max instances=1,
# 1 slot), so ~0.08 rows/s regardless of WORKERS. Full 963x4 would be ~15h+.
# Deterministic 400-row subsample (same rows for every judge) keeps it ~6h while
# the committed full-963 run 026 still carries the headline significance.
export ICEM_JUDGE_SAMPLE="${ICEM_JUDGE_SAMPLE:-400}"
WORKERS="${ICEM_JUDGE_WORKERS:-8}"

MODELS=(
  "qwen/qwen3.5-9b"      # protocol-matched control vs committed plain-completion run 026
  "gemma-4-26b-a4b-it"   # Google, big MoE -- cross-vendor
  "openai/gpt-oss-20b"   # OpenAI, reasoning -- max vendor diversity
)

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$PROG"; }

log "=== OVERNIGHT PANEL START (workers=$WORKERS) ==="
for m in "${MODELS[@]}"; do
  slug="${m//\//__}"; slug="${slug//:/_}"
  done_rows=0
  [ -f "$BASE/$slug/checkpoint.jsonl" ] && done_rows=$(wc -l < "$BASE/$slug/checkpoint.jsonl")
  log "JUDGE start: $m (resume from $done_rows rows)"
  if ICEM_JUDGE_MODEL="$m" $RUN python scripts/run_label_recovery_judge.py 0 "$WORKERS" >>"$PROG" 2>&1; then
    if $RUN python scripts/analyze_label_recovery.py "$BASE/$slug" "$m (tool-call)" >/dev/null 2>>"$PROG"; then
      log "JUDGE done + analyzed: $m"
    else
      log "!!! ANALYZE FAILED: $m"
    fi
  else
    log "!!! JUDGE FAILED: $m (skipping)"
  fi
done

log "=== AGGREGATE ==="
$RUN python scripts/aggregate_label_recovery_judges.py >>"$PROG" 2>&1 && log "aggregate written" || log "!!! AGGREGATE FAILED"

# Stop touching the server so JIT idle-TTL reclaims VRAM.
log "=== PANEL COMPLETE -- no further requests; JIT TTL will unload models ==="
touch "$RESULTS/PANEL_DONE.marker"
