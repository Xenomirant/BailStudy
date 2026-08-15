#!/bin/bash
# Overnight driver: sequential full-BailBench RB runs, then analysis + final sync.
# Separate python processes per model so GPU memory is fully released between runs.
set -uo pipefail
source /venv/main/bin/activate
[ -f /workspace/.env ] && set -a && source /workspace/.env && set +a  # HF_TOKEN etc.
cd /workspace/BailStudy

python -m rb_estimator.run_full_bench --model Qwen/Qwen2.5-7B-Instruct --t 768 --batch-size 192
q1=$?
python -m rb_estimator.run_full_bench --model google/gemma-2-9b-it --t 768 --batch-size 64
q2=$?
# retry once at half batch on failure (e.g. OOM); resume skips finished chunks
if [ $q1 -ne 0 ]; then
  python -m rb_estimator.run_full_bench --model Qwen/Qwen2.5-7B-Instruct --t 768 --batch-size 96
fi
if [ $q2 -ne 0 ]; then
  python -m rb_estimator.run_full_bench --model google/gemma-2-9b-it --t 768 --batch-size 32
fi
python -m rb_estimator.analyze || true
git add rb_estimator && git commit -q -m "rb-estimator: overnight results

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" && git push -q
echo "OVERNIGHT_COMPLETE"
