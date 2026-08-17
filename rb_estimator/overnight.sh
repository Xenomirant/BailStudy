#!/bin/bash
# Overnight driver: sequential full-BailBench RB runs, then analysis + final sync.
# Separate python processes per model so GPU memory is fully released between runs.
set -uo pipefail
source /venv/main/bin/activate
cd /workspace/BailStudy

run_model () {  # run_model <model> <t> <bs> <bs_retry>
  python -m rb_estimator.run_full_bench --model "$1" --t "$2" --n-traj 8 --batch-size "$3" \
    || python -m rb_estimator.run_full_bench --model "$1" --t "$2" --n-traj 8 --batch-size "$4"
}

run_model Qwen/Qwen2.5-7B-Instruct 768 192 96

# Night order (user): gemma-3-12b-it (pre-cached by user, no token needed),
# then Qwen3-8B thinking.
run_model google/gemma-3-12b-it 768 32 16
run_model Qwen/Qwen3-8B 1536 96 48
python -m rb_estimator.analyze || true
git add rb_estimator && git commit -q -m "rb-estimator: overnight results

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" && git push -q
echo "OVERNIGHT_COMPLETE"
