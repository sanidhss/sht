#!/bin/bash
# A (edge audit, CPU) -> B (inference array, afterok:A) -> C (collate, afterany:B).
# The layout is frozen before submitting: try06 died because a reorg happened while jobs
# sat in the queue.
set -e
D=/projects/bisa/sshah21/library/03_txgnn/try26_25-11-2026_unblinded-ceiling
A=$(sbatch --parsable $D/A_edge_audit.sb)
B=$(sbatch --parsable --dependency=afterok:$A $D/B_infer.sb)
C=$(sbatch --parsable --dependency=afterany:$B $D/C_collate.sb)
echo "A=$A B=$B C=$C"
