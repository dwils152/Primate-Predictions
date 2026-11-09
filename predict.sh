#!/bin/bash
# Check a genome and submit a GPU job that predicts CRMs across all of it.
#
#     ./predict.sh mPanTro3                 (a name from genomes.tsv)
#     ./predict.sh path/to/genome.fa        (any uncompressed FASTA file)
#
# Extra options are passed on to src/predict_genome.py, for example
#     ./predict.sh mPanTro3 --name chimp_chr1_test --contigs chr1_hap1_hsa1
set -euo pipefail
cd "$(dirname "$0")"
source scripts/common.sh
need_setup

if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
  echo "Usage: ./predict.sh <genome name or FASTA path> [options]"
  echo
  echo "Genomes listed in genomes.tsv:"
  cut -f1,3 genomes.tsv | tail -n +2 | sed 's/^/    /'
  echo
  echo "Options: $PY src/predict_genome.py --help"
  exit 0
fi

# Check everything on the login node first, so mistakes show up now instead
# of after waiting in the GPU queue.
"$PY" src/predict_genome.py "$@" --dry-run

job_id=$(sbatch --parsable scripts/predict.sbatch "$@")
explain_job "$job_id" "logs/predict-$job_id.out"
