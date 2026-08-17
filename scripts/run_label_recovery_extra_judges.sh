#!/usr/bin/env bash
# Multi-judge robustness for run 026: replicate the label/target-recovery judge
# across independent local judges via LM Studio TOOL CALLING (forces a clean
# structured answer; unlocks reasoning-tuned models whose plain-completion output
# lands in `reasoning_content`). Sequential: LM Studio serves one model at a time;
# concurrent models thrash VRAM. Each judge -> by_model/<slug>/{checkpoint,summary}.
#
# Panel: qwen3.5-9b is the primary judge re-run under the SAME tool protocol
# (protocol-matched control vs the committed plain-completion run 026); gpt-oss-20b
# (OpenAI) and gemma-4-26b (Google) add cross-vendor judges.
set -uo pipefail   # NOT -e: one bad judge must not abort the rest of the panel
cd "$(dirname "$0")/.." || exit 1

RUN="micromamba run -n icem-research"
BASE="docs/paper_woah_2026/results/026_label_recovery_judge/by_model"
export ICEM_JUDGE_TOOLS=1 KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1

# WORKERS=1 by default: LM Studio JIT spawns a SEPARATE model instance per
# concurrent request (qwen3.5-9b -> :2 :3 ... -> N copies in VRAM). Serial is the
# only safe default. To go faster, set LM Studio per-model "Max instances = 1"
# (queues concurrent requests on one instance), THEN raise WORKERS here.
WORKERS="${ICEM_JUDGE_WORKERS:-1}"

MODELS=(
  "qwen/qwen3.5-9b"
  "gemma-4-26b-a4b-it"
  "openai/gpt-oss-20b"
)

for m in "${MODELS[@]}"; do
  slug="${m//\//__}"; slug="${slug//:/_}"
  echo "=== JUDGE: $m  ($(date)) ==="
  if ICEM_JUDGE_MODEL="$m" $RUN python scripts/run_label_recovery_judge.py 0 "$WORKERS"; then
    echo "=== ANALYZE: $m ==="
    $RUN python scripts/analyze_label_recovery.py "$BASE/$slug" "$m (tool-call)" \
      || echo "!!! ANALYZE FAILED: $m"
  else
    echo "!!! JUDGE FAILED: $m — skipping"
  fi
  echo "=== DONE: $m  ($(date)) ==="
done
echo "ALL EXTRA JUDGES DONE"
