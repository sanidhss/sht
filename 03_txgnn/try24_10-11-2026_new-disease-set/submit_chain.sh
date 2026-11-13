#!/bin/bash
# Submits A -> B (array, afterok:A) -> C (afterany:B). Run from anywhere.
set -e
D=/projects/bisa/sshah21/library/03_txgnn/try24_10-11-2026_new-disease-set
A=$(sbatch --parsable $D/A_resolve_audit.sb)
B=$(sbatch --parsable --dependency=afterok:$A $D/B_train.sb)
C=$(sbatch --parsable --dependency=afterany:$B --export=ALL,B_JOB=$B $D/C_collate.sb)
echo "A=$A B=$B C=$C"
