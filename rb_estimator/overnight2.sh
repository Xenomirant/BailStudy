#!/bin/bash
# Revised night queue (user): finish Qwen2.5 map -> out-of-taxonomy multi-turn
# WildChat compounding -> Qwen3-14B (recent high-capability) full bench.
set -uo pipefail
source /venv/main/bin/activate
cd /workspace/BailStudy

python -m rb_estimator.run_full_bench --model Qwen/Qwen2.5-7B-Instruct --t 768 --n-traj 8 --batch-size 192 \
  || python -m rb_estimator.run_full_bench --model Qwen/Qwen2.5-7B-Instruct --t 768 --n-traj 8 --batch-size 96

python -m rb_estimator.run_multiturn --model Qwen/Qwen2.5-7B-Instruct --n-convs 500 --t 512 --n-traj 4 \
  || python -m rb_estimator.run_multiturn --model Qwen/Qwen2.5-7B-Instruct --n-convs 500 --t 512 --n-traj 4

python -m rb_estimator.run_full_bench --model Qwen/Qwen3-14B --t 1536 --n-traj 4 --batch-size 48 \
  || python -m rb_estimator.run_full_bench --model Qwen/Qwen3-14B --t 1536 --n-traj 4 --batch-size 24

python -m rb_estimator.analyze || true
git add rb_estimator && git commit -q -m "rb-estimator: overnight results

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" && git push -q
echo "OVERNIGHT_COMPLETE"
