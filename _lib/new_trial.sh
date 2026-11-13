#!/bin/bash
# Creates a new numbered try folder under library/<project>/, snapshots the given
# script(s) into it (the EXACT version about to run -- this is what solves the
# "intermediate versions get overwritten in place" problem hit throughout the manual
# reorg), and writes a README.md skeleton. Prints ONLY the new trial folder's absolute
# path to stdout, so a calling .sb script can capture it with $(...); everything else
# goes to stderr.
#
# Usage: new_trial.sh <project_dir> <short-kebab-description> <file1> [file2 ...]
#   project_dir: e.g. "03_txgnn" -- must already exist under library/
#   file1, file2, ...: the script(s) (.py/.sb/etc) to snapshot into the trial folder
#
# Pattern for use inside a .sb file:
#   TRIAL_DIR=$(/projects/bisa/sshah21/library/_lib/new_trial.sh 03_txgnn "some-fix" \
#       /path/to/script.py /path/to/this.sb)
#   python "$TRIAL_DIR/script.py"        # run the SNAPSHOT, not the original
#   /projects/bisa/sshah21/library/_lib/finalize_trial.sh "$TRIAL_DIR" "$SLURM_JOB_ID"
set -euo pipefail

LIBRARY_DIR="/projects/bisa/sshah21/library"

if [ "$#" -lt 3 ]; then
  echo "Usage: new_trial.sh <project_dir> <description> <file1> [file2 ...]" >&2
  exit 1
fi

PROJECT="$1"; shift
DESC="$1"; shift

PROJECT_DIR="$LIBRARY_DIR/$PROJECT"
if [ ! -d "$PROJECT_DIR" ]; then
  echo "ERROR: project dir $PROJECT_DIR does not exist -- create it (with a README.md)" \
       "before calling new_trial.sh, or check the project name." >&2
  exit 1
fi

LAST=$(find "$PROJECT_DIR" -maxdepth 1 -type d -name "try*" -printf '%f\n' 2>/dev/null \
       | sed -nE 's/^try([0-9]+)_.*/\1/p' | sort -n | tail -1)
if [ -z "$LAST" ]; then NEXT=1; else NEXT=$((10#$LAST + 1)); fi
NUM=$(printf "%02d" "$NEXT")
DATE=$(date +%d-%m-%Y)
TRIAL_DIR="$PROJECT_DIR/try${NUM}_${DATE}_${DESC}"

mkdir -p "$TRIAL_DIR"

for f in "$@"; do
  if [ ! -f "$f" ]; then
    echo "WARNING: $f not found, skipping snapshot of it" >&2
    continue
  fi
  cp "$f" "$TRIAL_DIR/"
done

TITLE="${DESC//-/ }"
{
  echo "# Try ${NUM} — ${TITLE}"
  echo
  echo "**Date:** ${DATE}"
  echo "**SLURM job:** ${SLURM_JOB_NAME:-<not running under slurm>} (${SLURM_JOB_ID:-<none>})"
  echo "**Scripts (exact versions run, snapshotted at submission time):**"
  for f in "$@"; do
    echo "- \`$(basename "$f")\`"
  done
  echo
  echo "## What it does"
  echo "TODO: fill in."
  echo
  echo "## What happened / why it failed"
  echo "TODO: the outcome section below is auto-filled by \`finalize_trial.sh\` with the"
  echo "real SLURM state and log file -- but the root-cause diagnosis, if it failed,"
  echo "still needs a human (or a coding assistant) to write, same as every prior try"
  echo "in this library."
} > "$TRIAL_DIR/README.md"

echo "$TRIAL_DIR"
