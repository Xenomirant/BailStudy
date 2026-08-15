#!/bin/bash
# Overnight driver: sequential full-BailBench RB runs, then analysis + final sync.
# Separate python processes per model so GPU memory is fully released between runs.
set -uo pipefail
source /venv/main/bin/activate
[ -f /workspace/.env ] && set -a && source /workspace/.env && set +a  # HF_TOKEN etc.
cd /workspace/BailStudy

run_model () {  # run_model <model> <t> <bs> <bs_retry>
  python -m rb_estimator.run_full_bench --model "$1" --t "$2" --n-traj 8 --batch-size "$3" \
    || python -m rb_estimator.run_full_bench --model "$1" --t "$2" --n-traj 8 --batch-size "$4"
}

run_model Qwen/Qwen2.5-7B-Instruct 768 192 96

# Night order (user): gemma first if its HF token arrives, then Qwen3-8B.
# gemma is gated: wait up to 3h for HF_TOKEN in /workspace/.env, then move on.
for i in $(seq 1 18); do
  [ -f /workspace/.env ] && set -a && source /workspace/.env && set +a
  [ -n "${HF_TOKEN:-}" ] && break
  echo "waiting for HF_TOKEN in /workspace/.env ($i/18)"; sleep 600
done
if [ -n "${HF_TOKEN:-}" ]; then
  run_model google/gemma-2-9b-it 768 64 32
else
  echo "NO_HF_TOKEN: skipping gemma-2-9b-it, going straight to Qwen3-8B"
fi

run_model Qwen/Qwen3-8B 1536 96 48
python -m rb_estimator.analyze || true
git add rb_estimator && git commit -q -m "rb-estimator: overnight results

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" && git push -q
echo "OVERNIGHT_COMPLETE"
