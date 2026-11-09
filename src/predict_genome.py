#!/usr/bin/env python3
"""Predict CRM probabilities at every base of a genome with the Human NTv3 CRM model.

Most people should not call this file directly. Use ``./predict.sh <genome>``,
which checks your input and submits this script as a GPU job.

What it does, step by step:

1. Checks that the model files in ``model_bundle/`` are the exact files this
   repository was tested with (SHA-256 checksums in ``bundle.sha256``).
2. Cuts every selected chromosome into 1,048,576-bp windows (the window size the
   model was trained on). The last window of a chromosome is padded with N.
3. Runs the model on each window on the GPU and keeps one probability per base.
4. Writes a per-base probability track (bigWig), predicted CRM regions (BED),
   a per-chromosome summary (TSV), and a provenance record (``run_info.json``).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = REPO / "model_bundle"
DEFAULT_RESULTS = REPO / "results"
GENOME_TABLE = REPO / "genomes.tsv"
LOCAL_GENOMES = REPO / "genomes"
INDEX_DIR = REPO / "genome_index"

# Fixed by the trained model. Do not change these.
WINDOW_BP = 1_048_576
# Region rule used when the model was evaluated: bases at or above the threshold
# are joined when separated by at most 50 bp, and regions shorter than 200 bp
# are dropped.
MERGE_GAP_BP = 50
MIN_REGION_BP = 200
# Measured on an A40 GPU when this repository was tested (see README).
SECONDS_PER_WINDOW_ESTIMATE = 2.5

FASTA_SUFFIXES = (".fa", ".fasta", ".fna")
ALLOWED_BASES = np.frombuffer(b"ACGTN", dtype=np.uint8)
NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


class UserError(Exception):
    """A problem the user can fix. Printed without a Python traceback."""


@dataclass(frozen=True)
class Genome:
    name: str
    fasta: Path
    species: str
    common_name: str
    source_url: str


@dataclass(frozen=True)
class Window:
    chrom: str
    start: int
    end: int


# --------------------------------------------------------------------------- #
# Finding the genome
# --------------------------------------------------------------------------- #


def read_genome_table() -> dict[str, dict[str, str]]:
    with GENOME_TABLE.open() as handle:
        rows = [row for row in csv.DictReader(handle, delimiter="\t") if row.get("name")]
    return {row["name"]: row for row in rows}


def resolve_genome(value: str, name: str | None) -> Genome:
    """Accept a name from genomes.tsv, a folder name under genomes/, or a FASTA path."""

    table = read_genome_table()
    path = Path(value)
    if value in table:
        row = table[value]
        genome = Genome(
            name=name or value,
            fasta=Path(row["fasta"]),
            species=row.get("species", ""),
            common_name=row.get("common_name", ""),
            source_url=row.get("source_url", ""),
        )
    elif (LOCAL_GENOMES / value / f"{value}.fa").is_file():
        genome = Genome(name or value, LOCAL_GENOMES / value / f"{value}.fa", "", "", "")
    elif path.suffix.lower() in FASTA_SUFFIXES or path.is_file():
        stem = path.name
        for suffix in FASTA_SUFFIXES:
            if stem.lower().endswith(suffix):
                stem = stem[: -len(suffix)]
        genome = Genome(name or stem, path, "", "", "")
    else:
        known = ", ".join(sorted(table)) or "(none)"
        raise UserError(
            f"I could not find a genome called {value!r}.\n"
            f"  Names in genomes.tsv: {known}\n"
            "  You can also give the path to a FASTA file, or download one with "
            "./download_genome.sh"
        )

    fasta = genome.fasta.resolve()
    if fasta.name.endswith((".gz", ".bgz")):
        raise UserError(
            f"{fasta} is compressed. Decompress it first, for example with\n"
            f"  gunzip -k {fasta}"
        )
    if not fasta.is_file():
        raise UserError(f"The genome file does not exist or I cannot read it: {fasta}")
    if not os.access(fasta, os.R_OK):
        raise UserError(f"You do not have permission to read {fasta}")
    if not NAME_PATTERN.match(genome.name):
        raise UserError(
            f"The run name {genome.name!r} may only contain letters, digits, '.', '_' and '-'"
        )
    source = genome.source_url
    source_json = fasta.parent / "source.json"
    if not source and source_json.is_file():
        source = json.loads(source_json.read_text()).get("url", "")
    return Genome(genome.name, fasta, genome.species, genome.common_name, source)


def fasta_index(fasta: Path) -> Path:
    """Return a .fai index for the FASTA, building one in genome_index/ if needed."""

    beside = Path(str(fasta) + ".fai")
    if beside.is_file() and beside.stat().st_mtime >= fasta.stat().st_mtime:
        return beside
    INDEX_DIR.mkdir(exist_ok=True)
    digest = hashlib.sha256(str(fasta).encode()).hexdigest()[:12]
    index = INDEX_DIR / f"{fasta.name}.{digest}.fai"
    if not index.is_file() or index.stat().st_mtime < fasta.stat().st_mtime:
        import pysam

        print(f"Indexing {fasta} (one time only, about a minute) ...", flush=True)
        temporary = index.with_suffix(".fai.partial")
        pysam.faidx(str(fasta), "--fai-idx", str(temporary))
        temporary.rename(index)
    return index


def read_fai(index: Path) -> list[tuple[str, int]]:
    contigs = []
    with index.open() as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            contigs.append((fields[0], int(fields[1])))
    return contigs


def select_contigs(
    contigs: list[tuple[str, int]], only: list[str] | None, min_bp: int
) -> tuple[list[tuple[str, int]], list[dict[str, Any]]]:
    if only:
        missing = sorted(set(only) - {name for name, _ in contigs})
        if missing:
            raise UserError(f"These contigs are not in the genome: {', '.join(missing)}")
    selected, excluded = [], []
    for name, length in contigs:
        if only and name not in only:
            excluded.append({"name": name, "length": length, "reason": "not requested"})
        elif length < min_bp:
            excluded.append({"name": name, "length": length, "reason": f"shorter than {min_bp} bp"})
        else:
            selected.append((name, length))
    if not selected:
        raise UserError("No contigs are left after filtering. Check --contigs and --min-contig-bp.")
    return selected, excluded


def make_windows(contigs: list[tuple[str, int]]) -> list[Window]:
    return [
        Window(name, start, min(start + WINDOW_BP, length))
        for name, length in contigs
        for start in range(0, length, WINDOW_BP)
    ]


# --------------------------------------------------------------------------- #
# Model bundle
# --------------------------------------------------------------------------- #


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_bundle(bundle: Path) -> dict[str, str]:
    """Check every bundle file against the checksums committed in bundle.sha256."""

    expected = {}
    for line in (REPO / "bundle.sha256").read_text().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            expected[name.strip()] = digest
    if not bundle.is_dir():
        raise UserError(f"The model bundle is missing: {bundle}\n  Run ./setup.sh first.")
    for name, digest in expected.items():
        path = bundle / name
        if not path.is_file():
            raise UserError(f"Model bundle file is missing: {path}\n  Run ./setup.sh again.")
        if sha256_file(path) != digest:
            raise UserError(
                f"Model bundle file has changed: {path}\n"
                "  Delete the model_bundle folder and run ./setup.sh again."
            )
    return expected


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #


def load_model(bundle: Path, device: Any) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from transformers import AutoTokenizer

    sys.path.insert(0, str(bundle / "code"))
    from NTv3CRMClassifier import NTv3CRMClassifier  # noqa: E402  (copied from the bundle)

    config = json.loads((bundle / "model" / "ntv3_crm_config.json").read_text())
    training = json.loads((bundle / "model" / "training_split.json").read_text())
    geometry = training["dataset_contract"]
    if int(geometry["input_length"]) != WINDOW_BP or int(geometry["target_length"]) != WINDOW_BP:
        raise RuntimeError("The bundle was trained with a different window size")
    model = NTv3CRMClassifier.from_crm_head(bundle / "model", map_location="cpu").to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"],
        revision=config["model_revision"],
        code_revision=config["model_code_revision"],
        trust_remote_code=True,
        padding_side="right",
    )
    # The model was trained and scored with fp16 autocast; use the same here.
    if training.get("bf16"):
        autocast_dtype = torch.bfloat16
    elif training.get("fp16"):
        autocast_dtype = torch.float16
    else:
        autocast_dtype = None
    info = {
        "model_name": config["model_name"],
        "model_revision": config["model_revision"],
        "model_code_revision": config["model_code_revision"],
        "autocast_dtype": str(autocast_dtype),
    }
    return model, tokenizer, {**info, "autocast_dtype": autocast_dtype}


def window_dataset(fasta: Path, index: Path, windows: list[Window], tokenizer: Any) -> Any:
    import torch
    from torch.utils.data import Dataset

    class GenomeWindows(Dataset):
        """Reads, cleans and tokenizes one window at a time (in worker processes)."""

        def __init__(self) -> None:
            self._fasta = None

        def __len__(self) -> int:
            return len(windows)

        def __getitem__(self, i: int) -> dict[str, Any]:
            import pysam

            if self._fasta is None:
                self._fasta = pysam.FastaFile(str(fasta), filepath_index=str(index))
            window = windows[i]
            sequence = self._fasta.fetch(window.chrom, window.start, window.end).upper()
            if len(sequence) != window.end - window.start:
                raise RuntimeError(f"Short read at {window.chrom}:{window.start}-{window.end}")
            codes = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
            unusual = ~np.isin(codes, ALLOWED_BASES)
            n_replaced = int(unusual.sum())
            if n_replaced:
                # IUPAC codes such as R or Y are not in the training data; treat them as N.
                codes = codes.copy()
                codes[unusual] = ord("N")
                sequence = codes.tobytes().decode("ascii")
            is_n = codes == ord("N")
            padded = sequence + "N" * (WINDOW_BP - len(sequence))
            encoded = tokenizer(
                padded,
                add_special_tokens=False,
                truncation=False,
                padding=False,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].squeeze(0).long()
            if input_ids.shape[0] != WINDOW_BP:
                raise RuntimeError("Tokenizer did not keep one token per base")
            return {
                "input_ids": input_ids,
                "index": torch.tensor(i),
                "is_n": torch.from_numpy(is_n.copy()),
                "n_replaced": torch.tensor(n_replaced),
            }

    return GenomeWindows()


def true_runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start and end (half-open) of every run of True values."""

    padded = np.concatenate(([False], mask, [False]))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return changes[0::2], changes[1::2]


def call_regions(
    probs: np.ndarray, is_n: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Merge above-threshold bases into regions with the evaluation rule."""

    values = probs.astype(np.float32)
    starts, ends = true_runs((values >= np.float32(threshold)) & ~is_n)
    if starts.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, np.zeros(0), np.zeros(0)
    new_region = np.concatenate(([True], starts[1:] > ends[:-1] + MERGE_GAP_BP))
    first = np.flatnonzero(new_region)
    last = np.concatenate((first[1:] - 1, [len(ends) - 1]))
    region_start, region_end = starts[first], ends[last]
    keep = region_end - region_start >= MIN_REGION_BP
    region_start, region_end = region_start[keep], region_end[keep]

    real = ~is_n
    cumulative = np.concatenate(([0.0], np.cumsum(np.where(real, values, 0.0), dtype=np.float64)))
    counts = np.concatenate(([0], np.cumsum(real, dtype=np.int64)))
    n_real = counts[region_end] - counts[region_start]
    mean = (cumulative[region_end] - cumulative[region_start]) / np.maximum(n_real, 1)
    peak = np.array(
        [values[s:e][real[s:e]].max() for s, e in zip(region_start, region_end)], dtype=np.float64
    )
    return region_start, region_end, mean, peak


class ContigWriter:
    """Writes the bigWig, BED and summary for one finished chromosome at a time."""

    def __init__(
        self, folder: Path, name: str, contigs: list[tuple[str, int]], threshold: float
    ) -> None:
        import pyBigWig

        if not pyBigWig.numpy:
            raise RuntimeError("pyBigWig was installed without numpy support")
        self.threshold = threshold
        self.name = name
        self.bigwig_path = folder / f"{name}.crm_probability.bw"
        self.bed_path = folder / f"{name}.crm_regions.bed"
        self.summary_path = folder / f"{name}.per_chromosome_summary.tsv"
        self.bigwig = pyBigWig.open(str(self.bigwig_path), "w")
        self.bigwig.addHeader([(chrom, int(length)) for chrom, length in contigs], maxZooms=10)
        self.bed = self.bed_path.open("w")
        self.summary_rows: list[dict[str, Any]] = []
        self.region_count = 0

    def write(self, chrom: str, probs: np.ndarray, is_n: np.ndarray, n_replaced: int) -> None:
        values = probs.astype(np.float32)
        starts, ends = true_runs(~is_n)
        chunk = 10_000_000
        for run_start, run_end in zip(starts, ends):
            for s in range(int(run_start), int(run_end), chunk):
                e = min(s + chunk, int(run_end))
                self.bigwig.addEntries(chrom, s, values=values[s:e], span=1, step=1)

        region_start, region_end, mean, peak = call_regions(probs, is_n, self.threshold)
        for s, e, m, p in zip(region_start, region_end, mean, peak):
            self.region_count += 1
            score = int(round(1000 * m))
            self.bed.write(
                f"{chrom}\t{s}\t{e}\t{self.name}_crm_{self.region_count:07d}\t{score}\t.\t"
                f"{m:.4f}\t{p:.4f}\n"
            )

        real = ~is_n
        n_real = int(real.sum())
        above = int(((values >= np.float32(self.threshold)) & real).sum())
        self.summary_rows.append(
            {
                "chrom": chrom,
                "length_bp": int(len(probs)),
                "scored_bp": n_real,
                "not_scored_bp": int(is_n.sum()),
                "unusual_bases_set_to_n": n_replaced,
                "mean_probability": round(float(values[real].mean()), 4) if n_real else "",
                "bp_above_threshold": above,
                "fraction_above_threshold": round(above / n_real, 4) if n_real else "",
                "crm_regions": int(len(region_start)),
                "crm_region_bp": int((region_end - region_start).sum()),
            }
        )

    def close(self) -> list[dict[str, Any]]:
        self.bigwig.close()
        self.bed.close()
        with self.summary_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(self.summary_rows[0]), delimiter="\t", lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(self.summary_rows)
        return self.summary_rows


def describe(seconds: float) -> str:
    hours, rest = divmod(int(seconds), 3600)
    return f"{hours} h {rest // 60:02d} min" if hours else f"{rest // 60} min {rest % 60:02d} s"


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run("status", "--porcelain", "--untracked-files=no")
    return {"commit": run("rev-parse", "HEAD"), "uncommitted_changes": bool(status)}


def predict(args: argparse.Namespace) -> Path:
    genome = resolve_genome(args.genome, args.name)
    index = fasta_index(genome.fasta)
    contigs, excluded = select_contigs(
        read_fai(index), args.contigs.split(",") if args.contigs else None, args.min_contig_bp
    )
    windows = make_windows(contigs)
    if args.max_windows:
        windows = windows[: args.max_windows]
        kept = {w.chrom for w in windows}
        contigs = [c for c in contigs if c[0] in kept]
    results = Path(args.results_dir).resolve()
    out = results / genome.name
    print(f"Genome      {genome.fasta}")
    print(f"Run name    {genome.name}")
    print(f"Contigs     {len(contigs)} selected, {len(excluded)} skipped")
    print(f"Windows     {len(windows):,} of {WINDOW_BP:,} bp")
    print(f"Estimate    about {describe(len(windows) * SECONDS_PER_WINDOW_ESTIMATE)} on one A40 GPU")
    print(f"Output      {out}")
    if out.exists() and not args.overwrite:
        raise UserError(
            f"Results for {genome.name!r} already exist in {out}\n"
            "  Choose another name with --name, or move/delete that folder first."
        )
    if args.dry_run:
        return out

    import torch

    if not torch.cuda.is_available():
        raise UserError(
            "No GPU is visible. Predictions must run on a GPU node.\n"
            "  Submit the job with ./predict.sh instead of running this file directly."
        )
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    bundle = Path(args.bundle).resolve()
    bundle_checksums = verify_bundle(bundle)
    threshold_record = json.loads((bundle / "human_threshold.json").read_text())
    threshold = float(threshold_record["threshold"])
    print("Checking the genome file checksum ...", flush=True)
    fasta_sha256 = sha256_file(genome.fasta)

    device = torch.device("cuda")
    model, tokenizer, model_info = load_model(bundle, device)
    autocast_dtype = model_info.pop("autocast_dtype")
    model_info["autocast_dtype"] = str(autocast_dtype)
    loader = torch.utils.data.DataLoader(
        window_dataset(genome.fasta, index, windows, tokenizer),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    partial = results / f"{genome.name}.partial"
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    writer = ContigWriter(partial, genome.name, contigs, threshold)
    lengths = dict(contigs)
    current: str | None = None
    probs = is_n = None
    n_replaced = 0
    print(f"Predicting {len(windows):,} window(s) ...", flush=True)
    loop_started = time.monotonic()
    try:
        with torch.inference_mode():
            for done, batch in enumerate(loader, start=1):
                window = windows[int(batch["index"])]
                if window.chrom != current:
                    if current is not None:
                        writer.write(current, probs, is_n, n_replaced)
                    current = window.chrom
                    probs = np.zeros(lengths[current], dtype=np.float16)
                    # True where there is no score: N bases, or bases a test run skipped.
                    is_n = np.ones(lengths[current], dtype=bool)
                    n_replaced = 0
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                if autocast_dtype is not None:
                    with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                        output = model(input_ids=input_ids)
                else:
                    output = model(input_ids=input_ids)
                alignment = model.last_output_alignment
                if alignment.input_start != 0 or alignment.input_end != WINDOW_BP:
                    raise RuntimeError("The model did not return one value per input base")
                length = window.end - window.start
                window_probs = torch.sigmoid(output["logits"])[0, :length]
                probs[window.start : window.end] = window_probs.cpu().numpy().astype(np.float16)
                is_n[window.start : window.end] = batch["is_n"][0].numpy()
                n_replaced += int(batch["n_replaced"])
                if done % 25 == 0 or done == len(windows):
                    rate = (time.monotonic() - loop_started) / done
                    print(
                        f"  [{done}/{len(windows)}] {window.chrom}:{window.start:,}  "
                        f"{rate:.2f} s/window, about {describe(rate * (len(windows) - done))} left",
                        flush=True,
                    )
        writer.write(current, probs, is_n, n_replaced)
        summary = writer.close()
    except BaseException:
        writer.bigwig.close()
        writer.bed.close()
        raise

    seconds_per_window = (time.monotonic() - loop_started) / len(windows)
    import peft
    import pyBigWig
    import pysam
    import transformers

    run_info = {
        "run_name": genome.name,
        "genome": {
            "fasta": str(genome.fasta),
            "fasta_sha256": fasta_sha256,
            "fasta_index": str(index),
            "species": genome.species,
            "common_name": genome.common_name,
            "source_url": genome.source_url,
        },
        "contigs_scored": [{"name": n, "length": l} for n, l in contigs],
        "contigs_skipped": excluded,
        "test_run_max_windows": args.max_windows,
        "model": {
            **model_info,
            "description": "NTv3 100M pretrained + LoRA rank 8, trained on Human (hg38) CRMs, seed 11",
            "bundle": str(bundle),
            "bundle_sha256": bundle_checksums,
        },
        "threshold": {
            "value": threshold,
            "chosen_on": "Human validation windows, per base, best F1",
            "region_merge_gap_bp": MERGE_GAP_BP,
            "region_min_bp": MIN_REGION_BP,
        },
        "window_bp": WINDOW_BP,
        "totals": {
            "windows": len(windows),
            "scored_bp": sum(row["scored_bp"] for row in summary),
            "crm_regions": writer.region_count,
            "crm_region_bp": sum(row["crm_region_bp"] for row in summary),
            "unusual_bases_set_to_n": sum(row["unusual_bases_set_to_n"] for row in summary),
        },
        "outputs": {
            "probability_bigwig": writer.bigwig_path.name,
            "regions_bed": writer.bed_path.name,
            "per_chromosome_summary": writer.summary_path.name,
        },
        "compute": {
            "started_utc": started_utc,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(time.monotonic() - started, 1),
            "seconds_per_window_including_writing": round(seconds_per_window, 3),
            "gpu": torch.cuda.get_device_name(device),
            "peak_gpu_memory_gib": round(torch.cuda.max_memory_allocated(device) / 2**30, 2),
            "host": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "pysam": pysam.__version__,
            "pyBigWig": getattr(pyBigWig, "__version__", "unknown"),
            "numpy": np.__version__,
            "repository": git_state(),
            "command": sys.argv,
        },
    }
    (partial / "run_info.json").write_text(json.dumps(run_info, indent=2) + "\n")
    if out.exists():
        shutil.rmtree(out)
    partial.rename(out)
    print(f"Done in {describe(time.monotonic() - started)}. Results are in {out}")
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("genome", help="a name from genomes.tsv, a folder in genomes/, or a FASTA path")
    parser.add_argument("--name", help="name for the results folder (default: the genome name)")
    parser.add_argument(
        "--contigs", help="comma-separated chromosome names to score (default: all long enough)"
    )
    parser.add_argument(
        "--min-contig-bp",
        type=int,
        default=1_000_000,
        help="skip contigs shorter than this (default: 1,000,000)",
    )
    parser.add_argument("--max-windows", type=int, help="score only the first N windows (for tests)")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true", help="replace existing results")
    parser.add_argument("--dry-run", action="store_true", help="check inputs and exit")
    return parser


def main(argv: list[str] | None = None) -> Path:
    args = build_parser().parse_args(argv)
    try:
        return predict(args)
    except UserError as error:
        print(f"\nERROR: {error}\n", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
