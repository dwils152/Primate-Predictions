#!/bin/bash
# Download a genome FASTA into genomes/<name>/<name>.fa with a record of where
# it came from. The download runs as a job on a compute node.
#
#     ./download_genome.sh rheMac10 https://hgdownload.soe.ucsc.edu/goldenPath/rheMac10/bigZips/rheMac10.fa.gz
#
# Afterwards, predict with:   ./predict.sh rheMac10
set -euo pipefail
cd "$(dirname "$0")"
source scripts/common.sh
need_setup

if [[ $# -ne 2 ]]; then
  echo "Usage: ./download_genome.sh <short name> <URL of a .fa, .fa.gz, .fasta or .fasta.gz file>"
  exit 1
fi
name="$1"
url="$2"
[[ "$name" =~ ^[A-Za-z0-9._-]+$ ]] || fail "Use only letters, digits, '.', '_' and '-' in the name."
[[ "$url" =~ ^https?:// ]] || fail "The second argument must be a web address (http or https)."
[[ ! -e "genomes/$name/$name.fa" ]] || fail "genomes/$name/$name.fa already exists."

job_id=$(sbatch --parsable scripts/download_genome.sbatch "$name" "$url")
explain_job "$job_id" "logs/download-$job_id.out"
