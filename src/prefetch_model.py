#!/usr/bin/env python3
"""Download the NTv3 base model into .hf_cache and check that it loads offline.

GPU jobs run with HF_HUB_OFFLINE=1, so everything must already be on disk.
Called by setup.sh. Loads the model once on the CPU; this takes a minute or two.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import predict_genome  # noqa: E402


def main() -> None:
    import torch

    offline = os.environ.get("HF_HUB_OFFLINE") == "1"
    bundle = predict_genome.DEFAULT_BUNDLE
    predict_genome.verify_bundle(bundle)
    model, tokenizer, info = predict_genome.load_model(bundle, torch.device("cpu"))
    ids = tokenizer("ACGTN", add_special_tokens=False)["input_ids"]
    if len(ids) != 5:
        raise RuntimeError("The tokenizer did not give one token per base")
    mode = "offline load works" if offline else "download finished"
    print(f"OK: {info['model_name']} at revision {info['model_revision'][:10]}, {mode}")


if __name__ == "__main__":
    main()
