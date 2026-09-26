#!/bin/bash
# Called at the END of a .sb script (after the main job command), regardless of that
# command's exit status. Copies the job's own SLURM log into its trial folder, appends
# the real outcome (state, elapsed, from sacct) to the README, then commits and pushes
# the whole library/ to GitHub. A failed push does NOT fail this script -- local commits
# still accumulate as a backup even if the remote is temporarily unreachable/unconfigured.
#
# Usage: finalize_trial.sh <trial_dir> <slurm_job_id>
set -uo pipefail  # intentionally not -e: git push failing shouldn't be fatal here

LIBRARY_DIR="/projects/bisa/sshah21/library"
TRIAL_DIR="$1"
JOB_ID="${2:-}"

if [ -n "$JOB_ID" ]; then
  LOG=$(find /work/hdd/bisa/sshah21/results -name "*_${JOB_ID}.out" 2>/dev/null | head -1)
  if [ -n "$LOG" ] && [ -f "$LOG" ]; then
    cp "$LOG" "$TRIAL_DIR/"
  fi

  # give sacct a moment -- accounting info can lag slightly behind job completion
  sleep 5
  INFO=$(sacct -j "$JOB_ID" --format=State,Elapsed -X --noheader 2>/dev/null | head -1 | xargs)
else
  INFO="unknown (no SLURM_JOB_ID passed)"
  LOG=""
fi

{
  echo
  echo "---"
  echo "## Auto-recorded outcome (finalize_trial.sh)"
  echo "**SLURM state / elapsed:** ${INFO:-unknown}"
  echo "**Log file:** $([ -n "$LOG" ] && basename "$LOG" || echo "<not found>")"
  echo
  echo "_If this failed, the root-cause diagnosis above this line still needs to be_"
  echo "_written by hand -- this section only records the raw outcome, not why._"
} >> "$TRIAL_DIR/README.md"

cd "$LIBRARY_DIR" || exit 0
git add -A
git commit -m "Auto-commit: job ${JOB_ID:-manual} finished (${INFO:-unknown})" --quiet \
  || echo "finalize_trial.sh: nothing to commit"
git push --quiet \
  || echo "finalize_trial.sh: WARNING -- git push failed, committed locally only. Check remote/auth."
