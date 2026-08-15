#!/bin/bash
# Overnight driver: sequential full-BailBench RB runs, then analysis + final sync.
# Separate python processes per model so GPU memory is fully released between runs.
set -uo pipefail
source /venv/main/bin/activate
[ -f /workspace/.env ] && set -a && source /workspace/.env && set +a  # HF_TOKEN etc.
cd /workspace/BailStudy

python -m rb_estimator.run_full_bench --model Qwen/Qwen2.5-7B-Instruct --t 768 --n-traj 8 --batch-size 192
q1=$?
# retry once at half batch on failure (e.g. OOM); resume skips finished chunks
if [ $q1 -ne 0 ]; then
  python -m rb_estimator.run_full_bench --model Qwen/Qwen2.5-7B-Instruct --t 768 --n-traj 8 --batch-size 96
fi

# gemma is gated: wait up to 5h for HF_TOKEN to appear in /workspace/.env
for i in $(seq 1 30); do
  [ -f /workspace/.env ] && set -a && source /workspace/.env && set +a
  [ -n "${HF_TOKEN:-}" ] && break
  echo "waiting for HF_TOKEN in /workspace/.env ($i/30)"; sleep 600
done
if [ -n "${HF_TOKEN:-}" ]; then
  python -m rb_estimator.run_full_bench --model google/gemma-2-9b-it --t 768 --n-traj 8 --batch-size 64
  q2=$?
  if [ $q2 -ne 0 ]; then
    python -m rb_estimator.run_full_bench --model google/gemma-2-9b-it --t 768 --n-traj 8 --batch-size 32
  fi
else
  echo "NO_HF_TOKEN: skipping gemma-2-9b-it"
fi
python -m rb_estimator.analyze || true
git add rb_estimator && git commit -q -m "rb-estimator: overnight results

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" && git push -q
echo "OVERNIGHT_COMPLETE"
