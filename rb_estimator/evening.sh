#!/bin/bash
# Evening queue: wait for the 14B bench to finish, then run the two new
# experiments on Qwen2.5: topic-consistent chains + trigger-string ablation.
set -uo pipefail
source /venv/main/bin/activate
cd /workspace/BailStudy

until [ -f rb_estimator/results/fullbench_qwen3-14b/DONE ]; do sleep 120; done
echo "14B_DONE_DETECTED"

python -m rb_estimator.run_chains --model Qwen/Qwen2.5-7B-Instruct
python -m rb_estimator.run_triggers --model Qwen/Qwen2.5-7B-Instruct
python -m rb_estimator.analyze || true
git add rb_estimator && git commit -q -m "rb-estimator: chains + trigger ablation results

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" && git push -q
echo "EVENING_COMPLETE"
