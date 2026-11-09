#!/bin/bash
# Submit a short GPU job that checks the installation against a known answer.
#
#     ./check_install.sh
set -euo pipefail
cd "$(dirname "$0")"
source scripts/common.sh
need_setup

job_id=$(sbatch --parsable scripts/check_install.sbatch)
explain_job "$job_id" "logs/check-install-$job_id.out"
echo
echo "It usually finishes within 10 minutes of starting. The last lines of the"
echo "log say PASS or FAIL."
