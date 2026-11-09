#!/usr/bin/env python3
"""Check the installation against a known answer.

Runs the full prediction pipeline on one 1-Mb window of the human genome
(hg38 chr10:8,388,608-9,437,184, a window the model never trained on) and
compares the result with the probabilities the original research pipeline
produced for the same window. Submit it with ./check_install.sh.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import predict_genome  # noqa: E402

# Different GPU models can round fp16 arithmetic slightly differently, so allow
# tiny differences. On the A40 GPU used for testing the match was exact.
MAX_ABS_DIFF = 0.01
MIN_CALL_AGREEMENT = 0.999


def main() -> None:
    import pyBigWig

    bundle = predict_genome.DEFAULT_BUNDLE
    fasta = bundle / "check" / "check_window.fa"
    out = predict_genome.main(
        [str(fasta), "--name", "install_check", "--min-contig-bp", "1", "--overwrite"]
    )
    expected = np.load(bundle / "check" / "expected_probs.npy").astype(np.float32)
    threshold = float(
        __import__("json").loads((bundle / "human_threshold.json").read_text())["threshold"]
    )
    with pyBigWig.open(str(out / "install_check.crm_probability.bw")) as bigwig:
        (record, length), = bigwig.chroms().items()
        observed = np.asarray(bigwig.values(record, 0, length, numpy=True), dtype=np.float32)

    if observed.shape != expected.shape or np.isnan(observed).any():
        print("FAIL: the output track does not cover the whole test window")
        raise SystemExit(1)
    diff = np.abs(observed - expected)
    agreement = float(((observed >= threshold) == (expected >= threshold)).mean())
    print()
    print(f"Bases compared            {len(expected):,}")
    print(f"Identical to reference    {float((diff == 0).mean()):.4%}")
    print(f"Largest difference        {float(diff.max()):.6f}  (allowed {MAX_ABS_DIFF})")
    print(f"Same CRM call (>= {threshold:.4f})  {agreement:.4%}  (needed {MIN_CALL_AGREEMENT:.1%})")
    if diff.max() <= MAX_ABS_DIFF and agreement >= MIN_CALL_AGREEMENT:
        print("\nPASS: the installation reproduces the reference predictions.")
    else:
        print("\nFAIL: the predictions differ from the reference. Ask for help before going on.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
