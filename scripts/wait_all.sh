#!/bin/bash
# Block until both the baseline job and the chained G3 job are finished.
while pgrep -f "run_experiments.py" > /dev/null || pgrep -f "chain_g3.sh" > /dev/null; do sleep 45; done
echo "ALL EXPERIMENT JOBS COMPLETE at $(date)"
