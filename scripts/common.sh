# Shared settings for every script in this repository. Do not run this file
# directly; the other scripts load it with "source".

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO/env/bin/python"

# Frozen copy of the trained model, readable by the zcsu_rsrch group.
BUNDLE_SOURCE="/projects/zcsu_research1/dwils152/crm-transformer-publication/code/results/handoff/primate_predictions_bundle_v1"

# Keep the ~400 MB base model inside this folder, not in your small home folder,
# but keep your private Hugging Face token in your home folder.
export HF_HOME="$REPO/.hf_cache"
export HF_TOKEN_PATH="$HOME/.cache/huggingface/token"
export TOKENIZERS_PARALLELISM=false
# Ignore Python packages installed in ~/.local or named by PYTHONPATH, which
# can break the environment.
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME

fail() {
  echo >&2
  echo "ERROR: $*" >&2
  echo >&2
  exit 1
}

need_setup() {
  [[ -x "$PY" && -d "$REPO/model_bundle" ]] || fail "Setup is not finished. Run ./setup.sh first."
}

# Print how to follow a submitted job.
explain_job() {
  local job_id="$1" log="$2"
  echo
  echo "Submitted job $job_id."
  echo "  Follow its progress:   tail -f $log      (press Ctrl+C to stop watching)"
  echo "  See if it is running:  squeue -u $USER"
  echo "  Cancel it:             scancel $job_id"
}
