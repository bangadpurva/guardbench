#!/bin/bash
# Wait for the in-flight baseline job, then run G3 single-pass on both API targets.
cd /Users/pbangad/guardbench
while pgrep -f "run_experiments.py --guardrails none G1" > /dev/null; do sleep 30; done
echo "=== baseline job finished, starting G3 at $(date) ==="
python3 run_experiments.py --guardrails G3 --models gpt4o_mini claude_haiku --runs 1
echo "=== G3 finished at $(date) ==="
