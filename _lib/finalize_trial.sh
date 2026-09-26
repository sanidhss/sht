#!/bin/bash
# Called at the END of a .sb script (after the main job command), regardless of that
# command's exit status. Copies the job's own SLURM log into its trial folder, appends
# the real outcome (state, elapsed, exit code, from sacct) plus a grep of common failure
# signatures to the README, then commits and pushes the whole library/ to GitHub. A
# failed push does NOT fail this script -- local commits still accumulate as a backup
# even if the remote is temporarily unreachable/unconfigured.
#
# The failure-signature grep is mechanical pattern matching, not a diagnosis -- it just
# surfaces the raw evidence (traceback, OOM, non-zero exit) so whoever writes the real
# root-cause explanation (human or Claude) doesn't have to go find it in the raw log.
#
# Usage: finalize_trial.sh <trial_dir> <slurm_job_id>
set -uo pipefail  # intentionally not -e: git push failing shouldn't be fatal here

LIBRARY_DIR="/projects/bisa/sshah21/library"
TRIAL_DIR="$1"
JOB_ID="${2:-}"

SIGNALS=""
if [ -n "$JOB_ID" ]; then
  LOG=$(find /work/hdd/bisa/sshah21/results -name "*_${JOB_ID}.out" 2>/dev/null | head -1)
  if [ -n "$LOG" ] && [ -f "$LOG" ]; then
    cp "$LOG" "$TRIAL_DIR/"
  fi

  # give sacct a moment -- accounting info can lag slightly behind job completion
  sleep 5
  INFO=$(sacct -j "$JOB_ID" --format=State,Elapsed,ExitCode -X --noheader 2>/dev/null | head -1 | xargs)

  if [ -n "$LOG" ] && [ -f "$LOG" ]; then
    SIGNALS=$(grep -E -A 15 'Traceback \(most recent call last\)' "$LOG" 2>/dev/null
              grep -E -i 'out of memory|CUDA error|RuntimeError|slurmstepd: error|srun: error|oom-kill|DUE TO TIME LIMIT|CANCELLED' "$LOG" 2>/dev/null)
  fi
else
  INFO="unknown (no SLURM_JOB_ID passed)"
  LOG=""
fi

{
  echo
  echo "---"
  echo "## Auto-recorded outcome (finalize_trial.sh)"
  echo "**SLURM state / elapsed / exit code:** ${INFO:-unknown}"
  echo "**Log file:** $([ -n "$LOG" ] && basename "$LOG" || echo "<not found>")"
  if [ -n "$SIGNALS" ]; then
    echo
    echo "### Auto-detected signals (grepped from log -- not a diagnosis)"
    echo '```'
    echo "$SIGNALS"
    echo '```'
  fi
  echo
  echo "_If this failed, the root-cause diagnosis above this line still needs to be_"
  echo "_written by hand -- this section only records the raw outcome, not why. See_"
  echo "_library/CLAUDE_INSTRUCTIONS.txt for who's expected to fill that in and when._"
} >> "$TRIAL_DIR/README.md"

cd "$LIBRARY_DIR" || exit 0
git add -A
git commit -m "Auto-commit: job ${JOB_ID:-manual} finished (${INFO:-unknown})

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>" --quiet \
  || echo "finalize_trial.sh: nothing to commit"
git push --quiet \
  || echo "finalize_trial.sh: WARNING -- git push failed, committed locally only. Check remote/auth."
