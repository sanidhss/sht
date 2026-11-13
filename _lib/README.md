# Trial automation

Two scripts, meant to be called from inside a `.sb` job script — one at the start, one
at the end — so every future job automatically gets its own numbered, dated try folder
with a README, and the whole library gets committed/pushed after every job finishes.

## `new_trial.sh` — call at the start of a job
```
TRIAL_DIR=$(/projects/bisa/sshah21/library/_lib/new_trial.sh <project> <description> <file1> [file2 ...])
```
- `<project>`: an existing project folder name, e.g. `03_txgnn`.
- `<description>`: short-kebab-case, becomes part of the folder name.
- `<file1> [file2 ...]`: the script(s) about to run (`.py`, `.sb`, etc).

Creates `library/<project>/tryNN_DD-MM-YYYY_<description>/`, auto-numbered from
whatever tries already exist in that project, **copies (snapshots) the given files into
it**, and writes a `README.md` skeleton (date, job name/ID if run under SLURM, which
scripts, `TODO` placeholders for "what it does" / "why it failed"). Prints only the new
folder's path to stdout.

**Then run the snapshot copy, not the original** — `python "$TRIAL_DIR/script.py"`, not
the original file path. This is what guarantees the exact version that ran is preserved
forever, which is the thing that couldn't be reconstructed for most of this library's
existing tries.

## `finalize_trial.sh` — call at the end of a job, unconditionally
```
/projects/bisa/sshah21/library/_lib/finalize_trial.sh "$TRIAL_DIR" "$SLURM_JOB_ID"
```
Copies the job's own `.out` log into the trial folder, appends the real SLURM outcome
(state, elapsed, via `sacct`) to the README, then `git add -A && git commit && git push`
from the library root. A failed push only prints a warning — it does not fail the
script, so local commits still accumulate as a backup even if the remote is
unreachable or not yet configured.

**The "why it failed" diagnosis is never automatic.** `finalize_trial.sh` records the
raw fact of what happened (state/exit code/log) and greps the log for common failure
signatures (traceback, OOM, non-zero exit, cancellation) as a labeled "not a
diagnosis" section — but writing the actual root-cause explanation still needs a human
or a coding assistant to read that evidence and fill in the `TODO` left by
`new_trial.sh`, same as every existing try in this library. (A local-only,
not-pushed-to-GitHub file at the library root has the fuller workflow instructions for
an assistant, if one is being used.)

## Full pattern for a `.sb` file
```bash
#!/bin/bash
#SBATCH ...

module load ...
source activate ...

TRIAL_DIR=$(/projects/bisa/sshah21/library/_lib/new_trial.sh 03_txgnn "some-fix" \
    /path/to/script.py "$0")
echo "Trial: $TRIAL_DIR"

/path/to/env/bin/python "$TRIAL_DIR/script.py"

/projects/bisa/sshah21/library/_lib/finalize_trial.sh "$TRIAL_DIR" "$SLURM_JOB_ID"
```
(`"$0"` snapshots the `.sb` file itself, since it's the second argument being passed in.)

## Status
Both scripts written and tested (dry run, no SLURM job) as of 25-09-2026. **Not yet
wired into any `.sb` file** — the 3 jobs in flight when this was built
(`22411998`/`22411999`/`22412000`) still use the pre-automation pattern; this applies
going forward. `git push` will fail (harmlessly, per above) until the library is
actually linked to a GitHub remote — see the top-level `library/README.md` for that
setup.
