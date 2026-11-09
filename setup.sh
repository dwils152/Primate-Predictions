#!/bin/bash
# One-time setup. Run it on the login node from inside this folder:
#
#     ./setup.sh
#
# It is safe to run again; finished steps are skipped.
set -euo pipefail
cd "$(dirname "$0")"
source scripts/common.sh

step() { echo; echo "==> $*"; }

step "1/5  Checking that you can read the shared model files"
if [[ ! -r "$BUNDLE_SOURCE/SHA256SUMS" ]]; then
  fail "I cannot read $BUNDLE_SOURCE
  Your account probably needs to be in the zcsu_rsrch group. Type 'groups' to
  see your groups, and ask your mentor to request access if it is missing."
fi
echo "OK"

step "2/5  Creating the Python environment in ./env (10-20 minutes the first time)"
if [[ -x "$PY" ]] && "$PY" -c "import torch, transformers, peft, pysam, pyBigWig" 2>/dev/null; then
  echo "Already done"
else
  rm -rf "$REPO/env"
  if ! command -v conda >/dev/null 2>&1; then
    # Make the cluster's "module" command and software list available, then
    # load the cluster's Anaconda. (Profile scripts are not written for set -u.)
    set +u
    type module >/dev/null 2>&1 || source /etc/profile.d/modules.sh
    [[ -f /etc/profile.d/modules-urc.sh ]] && source /etc/profile.d/modules-urc.sh
    module load anaconda3/2023.09
    set -u
    command -v conda >/dev/null 2>&1 || fail "Could not load the cluster's Anaconda module."
  fi
  # Keep conda's and pip's download caches out of your home folder.
  export CONDA_PKGS_DIRS="$REPO/.conda_pkgs"
  export PIP_NO_CACHE_DIR=1
  conda create --yes --quiet --prefix "$REPO/env" --override-channels -c conda-forge \
    python=3.13.8 pip
  "$PY" -m pip install --quiet -r requirements.txt
  rm -rf "$REPO/.conda_pkgs"
  "$PY" -c "import torch, transformers, peft, pysam, pyBigWig" ||
    fail "The environment was created but a package does not import. Ask for help."
  echo "OK"
fi

step "3/5  Copying the trained model into ./model_bundle"
if [[ ! -d model_bundle ]]; then
  rm -rf model_bundle.partial
  cp -r "$BUNDLE_SOURCE" model_bundle.partial
  mv model_bundle.partial model_bundle
fi
(cd model_bundle && sha256sum --quiet --check ../bundle.sha256) ||
  fail "The model files do not match bundle.sha256. Delete model_bundle and run ./setup.sh again."
echo "OK (checksums match)"

step "4/5  Logging in to Hugging Face"
if "$REPO/env/bin/hf" auth whoami >/dev/null 2>&1; then
  echo "Already logged in as $("$REPO/env/bin/hf" auth whoami | head -1)"
else
  cat <<'EOF'
The NTv3 base model is free but you must accept its terms first (see README,
"Before you start"). Then paste your Hugging Face access token below. The
token is not shown while you paste. Answer "n" if asked about git credentials.
EOF
  "$REPO/env/bin/hf" auth login
fi

step "5/5  Downloading the NTv3 base model (~400 MB) and testing it"
"$PY" src/prefetch_model.py
HF_HUB_OFFLINE=1 "$PY" src/prefetch_model.py

echo
echo "Setup finished. Next, test the installation on a GPU:"
echo "    ./check_install.sh"
