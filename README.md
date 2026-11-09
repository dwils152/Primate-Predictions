# Primate-Predictions

Predict cis-regulatory modules (CRMs) across whole primate genomes with a deep
learning model that was trained on human CRMs.

This guide is written for someone who has never used a computing cluster and is
new to research. Follow the sections in order the first time. Every command is
meant to be copied and pasted exactly as shown, unless the text tells you to
change part of it (for example, replacing `abc123` with your own username).
Inside a code box, lines that start with `#` are comments that explain the next
command.

**Short version**, once everything is set up

```bash
cd /projects/zcsu_research1/$USER/Primate-Predictions
./predict.sh mPanTro3
```

**How long things take**

| Step | Time | How often |
| --- | --- | --- |
| Get a cluster account and group access | a few days (someone else approves it) | once |
| Create a Hugging Face account and token | 10 minutes | once |
| `./setup.sh` | 5 to 20 minutes | once |
| `./check_install.sh` | about 10 minutes, mostly waiting for a GPU | once |
| `./predict.sh` for one genome | about 2 hours, plus waiting for a GPU | once per genome |

## Contents

1. [What this does](#1-what-this-does)
2. [Words you will see](#2-words-you-will-see)
3. [Terminal basics](#3-terminal-basics)
4. [Before you start (one time)](#4-before-you-start-one-time)
5. [Log in to the cluster](#5-log-in-to-the-cluster)
6. [Get the code (one time)](#6-get-the-code-one-time)
7. [Set up (one time)](#7-set-up-one-time)
8. [Test the installation (one time)](#8-test-the-installation-one-time)
9. [Predict CRMs in a primate genome](#9-predict-crms-in-a-primate-genome)
10. [Your results](#10-your-results)
11. [Chromosome names](#11-chromosome-names)
12. [Looking at the results](#12-looking-at-the-results)
13. [Other genomes](#13-other-genomes)
14. [What the predictions mean, and what they do not](#14-what-the-predictions-mean-and-what-they-do-not)
15. [Good research habits](#15-good-research-habits)
16. [Keeping the code up to date](#16-keeping-the-code-up-to-date)
17. [Disk space](#17-disk-space)
18. [Troubleshooting](#18-troubleshooting)
19. [Slurm cheat sheet](#19-slurm-cheat-sheet)
20. [Files in this repository](#20-files-in-this-repository)
21. [For maintainers](#21-for-maintainers)

## 1. What this does

A **CRM** (cis-regulatory module) is a stretch of DNA, such as an enhancer or a
promoter, that helps control when and where nearby genes are switched on.

The model reads DNA one million bases at a time and gives **every base** a
score between 0 and 1. A high score means "this looks like the human CRMs the
model was trained on". The model was trained on the human genome (hg38). Here
you apply it to other primates, which is called *cross-species prediction*.

For each genome you get

- a **probability track**, with one score per base, that you can view in a
  genome browser,
- a list of **predicted CRM regions**, which are stretches of bases that score
  above a fixed cutoff,
- a **summary table** with one line per chromosome, and
- a **record of the run** (`run_info.json`) that says exactly which files,
  model, settings and software produced the results.

The model is a large neural network (NTv3, 100 million parameters, made by
InstaDeep) plus a small part trained in our lab. It needs a GPU, so predictions
run as jobs on the cluster's GPU nodes. The scripts in this repository take
care of the details.

## 2. Words you will see

| Word | Meaning |
| --- | --- |
| HPC, cluster | Many computers that the university shares for research. Ours is called Starlight. |
| Terminal, shell | The window where you type commands. |
| Login node | The computer you land on when you log in. Use it only for small tasks such as editing files, submitting jobs and looking at results. |
| Compute node, GPU node | Computers that do the heavy work. You reach them by submitting a job. |
| GPU | A graphics card. Neural networks run much faster on one. |
| Slurm | The program that runs everyone's jobs on the compute nodes in turn. |
| Job | A script that Slurm runs for you. It waits in a queue until a node is free. Each job gets a number, the job ID. |
| Partition | A group of nodes. We use `GPU` for predictions and `Orion` for downloads. |
| Repository, repo | A folder of code tracked with git. This README belongs to one. |
| git, clone | git keeps track of versions of code. Cloning makes your own copy of a repository. |
| Environment | A private set of Python packages for this project, in the `env/` folder. |
| Path | Where a file lives, such as `/projects/zcsu_research1/abc123/Primate-Predictions/README.md`. |
| `$USER` | Shorthand the terminal replaces with your username. |
| Assembly | The DNA sequence of a species' genome, as a FASTA file. |
| T2T | Telomere-to-telomere, a genome assembly with every chromosome complete from end to end. |
| Haplotype | One of the two copies of each chromosome (one from each parent). |
| FASTA | A text file with DNA sequences (`.fa`, `.fasta`). |
| BED | A text file listing genomic regions, one per line (chromosome, start, end, ...). |
| bigWig, bedGraph | Files with a number for each position of a genome. bigWig is compressed; bedGraph is plain text. |
| Checksum (SHA-256) | A fingerprint of a file. If the file changes at all, the fingerprint changes. |

## 3. Terminal basics

You type a command and press Enter. These are the ones you need.

| Command | What it does |
| --- | --- |
| `pwd` | Shows which folder you are in. |
| `ls` | Lists the files in this folder. `ls -lh` also shows sizes. |
| `cd some_folder` | Goes into a folder. `cd ..` goes back up one level. |
| `less some_file` | Shows a text file one screen at a time. Arrow keys scroll; `q` quits. |
| `head some_file`, `tail some_file` | Show the first or last 10 lines of a file. |
| `rm some_file` | Deletes a file **permanently**. There is no trash folder on the cluster. |

Useful keys

- **Tab** completes file and folder names. Type the first few letters and press
  Tab.
- **Up arrow** brings back your previous commands.
- **Ctrl+C** stops the command that is running. Do not press it during
  `./setup.sh` unless you mean to stop it.
- **Pasting.** Right-clicking pastes in most terminals. Ctrl+V works in Windows
  PowerShell; on a Mac use Cmd+V; in Linux terminals use Ctrl+Shift+V.
- When a command asks for a password or token, nothing appears on screen as you
  type or paste. That is normal; press Enter when done.

## 4. Before you start (one time)

You need four things. Ask your mentor if any of them is a problem.

1. **A cluster account.** Request one on the
   [Research Computing "Get Started" page](https://oneit.charlotte.edu/urc/get-started).
   Students need a faculty sponsor; ask your mentor which faculty member to
   name.
2. **Membership of the lab's group, `zcsu_rsrch`.** This lets you read the lab's
   shared files, including the trained model and the primate genomes. Ask your
   mentor how to be added.
3. **The university VPN**, if you work off campus. The
   [IT help article on getting started with HPC](https://services.help.charlotte.edu/TDClient/33/Portal/KB/ArticleDet?ID=654)
   explains how to connect.
4. **A free Hugging Face account**, to download the NTv3 base model.
   1. Create an account at <https://huggingface.co/join>.
   2. While logged in, open both pages below and accept the terms on each. You
      get access right away.
      - <https://huggingface.co/InstaDeepAI/NTv3_100M_pre>
      - <https://huggingface.co/InstaDeepAI/ntv3_base_model>
   3. Create an access token at <https://huggingface.co/settings/tokens>. Click
      **Create new token**, choose the **Read** type, give it any name, and
      copy the token somewhere safe, such as a password manager. Treat it like
      a password. Never put it in a file in this repository or share it.

## 5. Log in to the cluster

Open a terminal on your own computer (Terminal on a Mac, PowerShell on
Windows). Type the line below, replacing `abc123` with your NinerNET username.

```bash
ssh abc123@hpc.charlotte.edu
```

- The first time, it asks `Are you sure you want to continue connecting
  (yes/no)?`. Type `yes` and press Enter.
- Enter your NinerNET password (nothing appears as you type) and approve the
  Duo prompt.

You can also get a terminal in your web browser through
[HPC OnDemand](https://oneit.charlotte.edu/innovation/research-computing/research-cluster/hpc-ondemand-user-guide/)
("Starlight Shell Access").

Check that you are in the lab's group

```bash
groups
```

The output must include `zcsu_rsrch`. If it does not, stop and ask your mentor.

To log out, type `exit`. Jobs you have submitted keep running after you log
out.

## 6. Get the code (one time)

Your home folder is small. This project needs about 7 GB for its software plus
about 10 GB for every genome you predict, so work in the lab's project space.

```bash
# Make your own folder in the lab's project space (does nothing if it exists)
mkdir -p /projects/zcsu_research1/$USER
cd /projects/zcsu_research1/$USER

# Download your own copy of this repository
git clone https://github.com/dwils152/Primate-Predictions.git
cd Primate-Predictions
```

From now on, **run every command from inside this `Primate-Predictions`
folder**. `pwd` should show `/projects/zcsu_research1/<your username>/Primate-Predictions`.
Whenever you log in again, start with

```bash
cd /projects/zcsu_research1/$USER/Primate-Predictions
```

## 7. Set up (one time)

Have your Hugging Face token ready (section 4), then run

```bash
./setup.sh
```

Keep the terminal open until it finishes (5 to 20 minutes). It works through
five steps, each starting with `==>`.

1. It checks that you can read the lab's shared model files.
2. It builds a private Python environment in `env/` with the exact package
   versions the model was tested with. This is the slow step. A `WARNING`
   about a "yanked version" of `transformers` is expected.
3. It copies the trained model into `model_bundle/` and checks every file
   against the checksums stored in `bundle.sha256`.
4. It asks for your Hugging Face token. Paste it and press Enter (nothing
   appears on screen). If it asks `Add token as git credential? (Y/n)`, type
   `n`. Your token is saved privately in your home folder, not in this
   repository.
5. It downloads the NTv3 base model (about 400 MB) and checks that it loads.

The end of a successful setup looks like this

```
==> 5/5  Downloading the NTv3 base model (~400 MB) and testing it
OK: InstaDeepAI/NTv3_100M_pre at revision 5c685dca15, download finished
OK: InstaDeepAI/NTv3_100M_pre at revision 5c685dca15, offline load works

Setup finished. Next, test the installation on a GPU:
    ./check_install.sh
```

If it stops with `ERROR`, read the message, fix what it names (section 18
lists common problems), and run `./setup.sh` again. Finished steps are
skipped, so running it again is always safe. If your connection drops during
setup, log in again, go back to the folder, and rerun it.

## 8. Test the installation (one time)

This runs the model on one piece of the human genome whose correct answer we
already know, and compares the two. Do this before predicting anything.

```bash
./check_install.sh
```

It prints something like

```
Submitted job 27712854.
  Follow its progress:   tail -f logs/check-install-27712854.out      (press Ctrl+C to stop watching)
  See if it is running:  squeue -u abc123
  Cancel it:             scancel 27712854
```

The job first waits in the queue until a GPU is free (often a few minutes,
sometimes longer). While it waits, the log file does not exist yet and
`squeue -u $USER` shows the job with state `PD` (pending). Once it runs (state
`R`), it takes about a minute. Then run the `tail -f` command it printed. The
log ends like this

```
Bases compared            1,048,576
Identical to reference    100.0000%
Largest difference        0.000000  (allowed 0.01)
Same CRM call (>= 0.4888)  100.0000%  (needed 99.9%)

PASS: the installation reproduces the reference predictions.
Finished Mon Nov  9 12:53:38 EST 2026
```

Press Ctrl+C to stop watching. If you see `FAIL`, or the job ends without
`PASS`, do not run predictions. Send your mentor the log file path.

## 9. Predict CRMs in a primate genome

These genomes are already on the cluster, so you do not need to download them.

| Name | Species | Common name |
| --- | --- | --- |
| `mPanTro3` | *Pan troglodytes* | chimpanzee |
| `mPanPan1` | *Pan paniscus* | bonobo |
| `mGorGor1` | *Gorilla gorilla* | gorilla |
| `mPonAbe1` | *Pongo abelii* | Sumatran orangutan |
| `mPonPyg2` | *Pongo pygmaeus* | Bornean orangutan |
| `mSymSyn1` | *Symphalangus syndactylus* | siamang |

They are complete, telomere-to-telomere assemblies from the
[T2T Primates project](https://github.com/marbl/Primates), downloaded from
GenomeArk, one haplotype each. `genomes.tsv` lists their file locations and
download addresses.

### A practice run (optional, about 5 minutes)

Before your first whole genome, you can score one small chimpanzee chromosome
to see what the outputs look like.

```bash
./predict.sh mPanTro3 --name chimp_practice --contigs chr22_hap1_hsa21
```

### A whole genome

```bash
./predict.sh mPanTro3
```

First the script checks your input and prints a plan. The first time you use a
genome it also spends about a minute indexing it.

```
Genome      /projects/zcsu_research1/Primate_Genomes/mPanTro3.pri.cur.20231122.fasta
Run name    mPanTro3
Contigs     25 selected, 0 skipped
Windows     3,040 of 1,048,576 bp
Estimate    about 2 h 06 min on one A40 GPU
Output      /projects/zcsu_research1/abc123/Primate-Predictions/results/mPanTro3

Submitted job 27712900.
  Follow its progress:   tail -f logs/predict-27712900.out      (press Ctrl+C to stop watching)
  See if it is running:  squeue -u abc123
  Cancel it:             scancel 27712900
```

Write the job number in your notebook (section 15). **You can log out while
the job runs.** The log shows progress every 25 windows, with an estimate of
the time left, for example

```
Predicting 3,040 window(s) ...
  [25/3040] chr1_hap1_hsa1:25,165,824  1.94 s/window, about 1 h 37 min left
  [50/3040] chr1_hap1_hsa1:51,380,224  1.92 s/window, about 1 h 35 min left
```

The job has finished successfully when the log ends with

```
Done in 2 h 01 min. Results are in /projects/zcsu_research1/abc123/Primate-Predictions/results/mPanTro3
Finished ...
```

To check on all your jobs from today, including finished ones

```bash
sacct -u $USER -S today -X --format=JobID,JobName,State,Elapsed
```

`COMPLETED` means it worked. `FAILED`, `TIMEOUT`, `OUT_OF_MEMORY` or
`CANCELLED` mean it did not (section 18).

### All six genomes

Each `./predict.sh` command becomes its own job, so you can submit several.
Either one at a time

```bash
./predict.sh mPanPan1
./predict.sh mGorGor1
```

or all six at once

```bash
for genome in mPanTro3 mPanPan1 mGorGor1 mPonAbe1 mPonPyg2 mSymSyn1; do
  ./predict.sh "$genome"
done
```

Skip any genome you have already predicted; the script refuses to overwrite
existing results.

### Options

Add these after the genome name.

| Option | What it does |
| --- | --- |
| `--name NAME` | Name of the results folder (default: the genome name). Use letters, digits, `.`, `_` and `-` only. |
| `--contigs A,B` | Score only these chromosomes (comma-separated, no spaces). See section 11 for the names. |
| `--min-contig-bp N` | Skip chromosomes and contigs shorter than N bases (default 1,000,000). |
| `--overwrite` | Replace existing results with the same name. Avoid this; use a new `--name` instead. |

## 10. Your results

Each run gets its own folder, `results/<name>/`.

| File | What it is |
| --- | --- |
| `<name>.crm_probability.bw` | Score (0 to 1) of every base. Bases that are `N` (unknown sequence) in the genome have no score. |
| `<name>.crm_regions.bed` | Predicted CRM regions (see below). |
| `<name>.per_chromosome_summary.tsv` | One line per chromosome (see below). |
| `run_info.json` | Exactly what produced these results (see below). **Keep it with the results.** |

### Predicted regions

A region is a run of bases with a score of at least **0.4888**. Runs separated by
50 bp or less are joined, and regions shorter than 200 bp are dropped. The
cutoff was chosen on held-out human data before any primate was scored, and
both rules are the ones used to evaluate the model in human. Do not change them
to make results look better (section 15).

Expect a large share of each chromosome to be called; it was 38% of the
chimpanzee chromosome used to test this repository. That follows from the human
training labels, in which CRMs make up a large share of the labeled bases.

The BED file has 8 columns, separated by tabs.

| Column | Meaning |
| --- | --- |
| 1 to 3 | Chromosome, start, end. Counting starts at 0 and the end base is not included, as in every BED file. |
| 4 | Region name. |
| 5 | Mean score of the region times 1000, rounded (genome browsers use this to shade regions). |
| 6 | Strand (always `.`, regions have no strand). |
| 7 | Mean score of the region. |
| 8 | Highest score in the region. |

### Per-chromosome summary

| Column | Meaning |
| --- | --- |
| `chrom`, `length_bp` | Chromosome name and length. |
| `scored_bp`, `not_scored_bp` | Bases that got a score, and `N` bases that did not. |
| `unusual_bases_set_to_n` | Letters other than A, C, G, T and N (such as R or Y), treated as N. Usually 0. |
| `mean_probability` | Mean score over the scored bases. |
| `bp_above_threshold`, `fraction_above_threshold` | Bases scoring at least 0.4888, as a count and a fraction. |
| `crm_regions`, `crm_region_bp` | Number of predicted regions and the bases they cover. |

### The run record

`run_info.json` lists the genome file and its checksum, the chromosomes scored
and skipped, the model files and their checksums, the cutoff and region rules,
totals, the GPU, the job number, run time, software versions, and the version
(git commit) of this repository. To read it

```bash
less results/mPanTro3/run_info.json
```

### Size

Each genome needs about 10 GB of disk space, almost all of it the probability
track. The BED file is about 70 MB.

## 11. Chromosome names

The T2T assemblies number each species' chromosomes roughly by size, so an ape
chromosome usually does not match the human chromosome with the same number.
For example, chimpanzee chromosome 2 matches human chromosome 3. The names say
how they match.

| Example | Meaning |
| --- | --- |
| `chr1_hap1_hsa1` | Chromosome 1, from haplotype 1, matching human (*Homo sapiens*, "hsa") chromosome 1. |
| `chr5_mat_hsa6` | Chromosome 5, the copy inherited from the mother (`mat`; `pat` is the father's), matching human chromosome 6. |
| `chr12_pat_hsa2a` | Human chromosome 2 is a fusion of two ape chromosomes, called 2a and 2b. |
| `chr4_pat_hsa17x5` | A gorilla chromosome made of parts that match human chromosomes 17 and 5. |
| `chr1_hap1` (siamang) | Gibbon chromosomes are so rearranged relative to human that the names carry no human match. |

To list the chromosome names of a genome (this takes up to a minute)

```bash
grep '^>' /projects/zcsu_research1/Primate_Genomes/mPanTro3.pri.cur.20231122.fasta
```

After a prediction, the first column of the summary file lists them too.

## 12. Looking at the results

### In the terminal

```bash
# The per-chromosome summary as a table
column -t results/mPanTro3/mPanTro3.per_chromosome_summary.tsv

# How many regions were predicted
wc -l results/mPanTro3/mPanTro3.crm_regions.bed

# The 10 regions with the highest mean score
sort -k7,7gr results/mPanTro3/mPanTro3.crm_regions.bed | head
```

Scores of single bases (this uses the Python environment made by setup)

```bash
env/bin/python -c '
import pyBigWig
bw = pyBigWig.open("results/mPanTro3/mPanTro3.crm_probability.bw")
print(bw.chroms())                                     # chromosome names and lengths
print(bw.stats("chr1_hap1_hsa1", 1000000, 1010000))    # mean score of a 10-kb stretch
print(bw.values("chr1_hap1_hsa1", 1000000, 1000010))   # scores of its first 10 bases
'
```

### In a genome browser (IGV)

The whole probability track is about 10 GB, which is a lot to download. It is
easier to cut out the region you want to look at.

**Step 1, on the cluster.** Edit the first line (run name, chromosome, start,
end) and paste the whole block. It writes a small bedGraph of the scores and a
BED file of the regions into the `exports/` folder.

```bash
RUN=mPanTro3; CHROM=chr1_hap1_hsa1; START=1000000; END=1100000
mkdir -p exports
env/bin/python - "$RUN" "$CHROM" "$START" "$END" <<'EOF'
import sys
from pathlib import Path
import pyBigWig

run, chrom, start, end = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
results = Path("results") / run
stem = Path("exports") / f"{run}_{chrom}_{start}_{end}"
with pyBigWig.open(str(results / f"{run}.crm_probability.bw")) as bigwig:
    with open(f"{stem}.bedGraph", "w") as out:
        for s, e, value in bigwig.intervals(chrom, start, end) or []:
            out.write(f"{chrom}\t{s}\t{e}\t{value:.4f}\n")
with open(results / f"{run}.crm_regions.bed") as bed, open(f"{stem}.bed", "w") as out:
    for line in bed:
        c, s, e = line.split("\t")[:3]
        if c == chrom and int(e) > start and int(s) < end:
            out.write(line)
print(f"Wrote {stem}.bedGraph and {stem}.bed")
EOF
```

IGV also needs the genome sequence and its index file (`.fai`), with the same
name plus `.fai`, in the same folder. The gorilla genome's index sits next to
its FASTA. For the other genomes the prediction made one in `genome_index/`;
copy it into `exports/` under the right name, for example for the chimpanzee

```bash
cp genome_index/mPanTro3.pri.cur.20231122.fasta.*.fai exports/mPanTro3.pri.cur.20231122.fasta.fai
```

**Step 2, on your own computer.** Open a new terminal on your computer (not the
one logged in to the cluster) and copy the files. Replace `abc123` with your
username. The genome is about 3 GB, so copy it only once.

```bash
# The exported scores and regions
scp "abc123@hpc.charlotte.edu:/projects/zcsu_research1/abc123/Primate-Predictions/exports/mPanTro3_*" .

# The genome and its index (once per genome)
scp abc123@hpc.charlotte.edu:/projects/zcsu_research1/Primate_Genomes/mPanTro3.pri.cur.20231122.fasta .
scp abc123@hpc.charlotte.edu:/projects/zcsu_research1/abc123/Primate-Predictions/exports/mPanTro3.pri.cur.20231122.fasta.fai .
```

The files land in the folder your terminal is in (`pwd` shows which). You can
also download files with the file browser in HPC OnDemand.

**Step 3, in IGV.** Install [IGV](https://igv.org/doc/desktop/). Load the
genome with **Genomes > Load Genome from File** (choose the `.fasta` file),
then load the `.bedGraph` and `.bed` files with **File > Load from File**. Type
the region, for example `chr1_hap1_hsa1:1000000-1100000`, in the box at the
top and press Enter.

## 13. Other genomes

You can use any genome FASTA file, for example a monkey genome from the
[UCSC Genome Browser downloads](https://hgdownload.soe.ucsc.edu/downloads.html).
Talk to your mentor before adding a species, and download it into `genomes/`
with a record of where it came from

```bash
./download_genome.sh rheMac10 https://hgdownload.soe.ucsc.edu/goldenPath/rheMac10/bigZips/rheMac10.fa.gz
```

This runs as a job (follow it with the `tail -f` command it prints). When the
log ends with `Done`, the genome is in `genomes/rheMac10/`, next to a
`source.json` file that records the address, date and checksums. Then predict
as usual

```bash
./predict.sh rheMac10
```

You can also give the path of any uncompressed FASTA file directly

```bash
./predict.sh /path/to/genome.fa
```

By default only chromosomes and contigs of at least 1,000,000 bp are scored.
That skips the many small unplaced pieces in most assemblies, and the
mitochondrial genome. Use `--min-contig-bp` to change it (section 9).

## 14. What the predictions mean, and what they do not

Read this before you interpret or present any result, and discuss your plans
with your mentor.

- **Predictions are not proof.** A high score means the sequence resembles the
  human CRMs the model learned from. Treat predicted regions as candidates to
  be tested against independent evidence.
- **The model only knows human.** It was trained on human DNA and human CRM
  labels. Applying it to another species assumes that regulatory DNA there
  looks like human regulatory DNA. The cutoff was also chosen in human, so its
  meaning in other species is unknown.
- **Close relatives look like human.** Chimpanzee and bonobo DNA is roughly 99%
  identical to human DNA where the genomes align. Predictions there will often
  just mirror the human answer. That is expected from the sequence similarity
  and does not show that the model has learned how regulation works.
- **Repeats matter.** In the human training data, most non-CRM bases are
  repetitive DNA and most CRM bases are not, so part of what the model learned
  is to tell repeats from non-repeat sequence. Before drawing conclusions,
  compare the predictions with a repeat annotation. The T2T assemblies here are
  not repeat-masked; ask your mentor which annotation to use.
- **Parts of these genomes were never seen in training.** The T2T assemblies
  include centromeres and other large satellite arrays that were gaps in the
  human reference used for training. Be careful with predictions there.
- **Comparing with known CRMs needs care.** Some human regions were used to
  train the model, and their primate counterparts are nearly identical, so
  agreement there is not a fair test. Talk to your mentor before any
  evaluation.

## 15. Good research habits

- **Keep a lab notebook.** A text file is fine, for example
  `nano ~/notebook.txt` (save with Ctrl+O then Enter, quit with Ctrl+X). For
  every run, note the date, the exact command, the job number, the result, and
  anything unusual.
- **Note the code version** at the start of each batch of runs with
  `git log -1 --oneline`. Results are also stamped with it in `run_info.json`.
- **Never edit files in `results/`.** If something needs to change, rerun with
  a new `--name`.
- **Keep each `run_info.json` with its results.** It is how anyone can later
  check exactly what produced them.
- **Do not change the cutoff, the region rules or the package versions** to get
  "better" results. Choosing settings after seeing the answer makes results
  unreliable. If you think a setting is wrong, ask first.
- **Write down failures too.** A job that failed, or a genome that gave strange
  results, is useful information.
- **Do not commit genomes, results or the model to git.** The `.gitignore`
  file already prevents it. This repository is public; the model is not.

After each genome, send your mentor the results folder path, the output of
`column -t results/<name>/<name>.per_chromosome_summary.tsv`, and anything
that looked odd.

## 16. Keeping the code up to date

When your mentor tells you the code has changed, update your copy with

```bash
git pull
```

`git pull` lists the files that changed.

- If `requirements.txt` is among them, delete the environment with
  `rm -rf env` and run `./setup.sh`.
- If `bundle.sha256` is among them, delete the model copy with
  `rm -rf model_bundle` and run `./setup.sh`.
- In either case, run `./check_install.sh` again.

If `git pull` complains about "local changes", you have edited a file in the
repository. Ask your mentor before doing anything else. Do not update in the
middle of a set of runs you plan to compare with each other.

## 17. Disk space

```bash
# Space used by each of your results folders
du -sh results/*

# Free space in the lab's project area
df -h /projects/zcsu_research1
```

Ask your mentor before deleting any results. `results/<name>.partial` folders
are left by jobs that stopped early and can be deleted with
`rm -r results/<name>.partial`; they are also removed automatically when you
rerun the same name.

## 18. Troubleshooting

| Message or symptom | What to do |
| --- | --- |
| `Permission denied` when running `./setup.sh` | Run `bash setup.sh` instead (the same works for the other scripts). |
| `No such file or directory` for `./setup.sh` or other scripts | You are not in the `Primate-Predictions` folder. Run `cd /projects/zcsu_research1/$USER/Primate-Predictions`. |
| A `WARNING` about a "yanked version" of `transformers` | Expected. The model was tested with that exact version, so setup installs it on purpose. |
| `Setup is not finished` | Run `./setup.sh`. |
| `I cannot read ... zcsu_rsrch` | You are not in the lab's group yet (section 4). |
| `401`, `403` or `gated repo` during setup step 5 | You have not accepted the terms on both Hugging Face pages, or the token is wrong. Accept the terms, run `env/bin/hf auth login`, then `./setup.sh` again. |
| `Disk quota exceeded` | You are probably working in your home folder. Move to `/projects/zcsu_research1/$USER` (section 6). |
| `tail: cannot open 'logs/...'` | The job has not started yet, so there is no log. Check with `squeue -u $USER` and try again later. |
| `squeue` shows `PD` with `(Resources)` or `(Priority)` | The job is waiting for a free GPU. This is normal; wait. |
| `I could not find a genome called ...` | Check the spelling against the table in section 9 (names are case-sensitive). |
| `Results for ... already exist` | Pick a new name with `--name`, or ask your mentor about the old folder. |
| `These contigs are not in the genome` | Check the chromosome names (section 11). Separate several with commas and no spaces. |
| The job disappears from `squeue` with no `Done` in the log | Look at the end of the log (`tail -50 logs/predict-<job>.out`) and run `sacct -j <job>`. `TIMEOUT` means it ran out of time; `OUT_OF_MEMORY` means it ran out of memory. Send the log path to your mentor. |
| `No GPU is visible` | You ran the Python script on the login node. Use `./predict.sh`. |

When you ask for help, always include the command you ran, the job number, and
the path of the log file. For problems with your account or the cluster itself
(not this project), contact University Research Computing through their
[Get Started page](https://oneit.charlotte.edu/urc/get-started).

## 19. Slurm cheat sheet

| Command | What it does |
| --- | --- |
| `squeue -u $USER` | Your jobs that are waiting (`PD`) or running (`R`). |
| `tail -f logs/<file>.out` | Follow a log as it grows. Ctrl+C stops watching (the job keeps running). |
| `scancel <job>` | Cancel a job. |
| `sacct -u $USER -S today -X --format=JobID,JobName,State,Elapsed` | All your jobs from today, finished or not. |
| `sacct -j <job> --format=JobID,State,Elapsed,MaxRSS` | How one job went. |
| `seff <job>` | How much time and memory a finished job used. |

More detail is in the cluster's
[ORION, Nebula and GPU user notes](https://oneit.charlotte.edu/urc/research-clusters/orion-gpu-slurm-user-notes).

## 20. Files in this repository

| Path | What it is |
| --- | --- |
| `README.md` | This guide. |
| `setup.sh` | One-time setup (section 7). |
| `check_install.sh` | Tests the installation on a GPU (section 8). |
| `predict.sh` | Checks a genome and submits a prediction job (section 9). |
| `download_genome.sh` | Downloads a genome as a job (section 13). |
| `genomes.tsv` | The genomes you can use by name: name, species, common name, file, download address. |
| `requirements.txt` | Exact Python package versions. |
| `bundle.sha256` | Checksums of the trained model files. |
| `scripts/` | The job scripts (`*.sbatch`) and shared settings (`common.sh`) the commands above use. |
| `src/predict_genome.py` | The prediction program. |
| `src/check_install.py`, `src/prefetch_model.py` | Used by the installation check and by setup. |
| `logs/` | Job logs, one per job. |

Folders created on your copy, and never committed to git: `env/` (Python
environment), `model_bundle/` (trained model), `.hf_cache/` (NTv3 base model),
`genome_index/` (genome index files), `genomes/` (downloaded genomes),
`results/` (predictions), `exports/` (regions cut out for viewing).

## 21. For maintainers

- **Model.** NTv3 100M pretrained (`InstaDeepAI/NTv3_100M_pre`, revision
  `5c685dca15891f5c5b80e0c930e23b87a217e441`, code revision
  `0ecff3637f0d3ba5b686d1095083218157c2ca34`) with a LoRA rank-8 adapter and a
  per-base MLP head, trained on Human (hg38) CRMs with seed 11.
- **Bundle.** The trained files are not in this public repository. `setup.sh`
  copies a frozen bundle from
  `/projects/zcsu_research1/dwils152/crm-transformer-publication/code/results/handoff/primate_predictions_bundle_v1`
  and checks it against `bundle.sha256`. The bundle's `PROVENANCE.json` lists
  the source of every file. It holds the adapter and head, the per-base
  threshold chosen on Human validation windows, a copy of the model class used
  to score the model (`NTv3CRMClassifier.py`), and a check window with its
  reference probabilities. To change the model, build a new bundle version and
  update `bundle.sha256` and `BUNDLE_SOURCE` in `scripts/common.sh`; never
  edit an existing bundle.
- **Prediction.** Contigs are tiled from position 0 in 1,048,576-bp windows,
  the last window padded with N. Input is uppercased, and characters other than
  `ACGTN` become N (counted in the summary). Inference uses fp16 autocast like
  training. These match the research scoring pipeline.
- **Verification (2026-11-09, on Starlight, from a shell without a personal
  conda).**
  - `setup.sh` built the environment from the cluster's Anaconda module and
    copied the bundle with matching checksums.
  - The installation check on an A40 reproduced the research pipeline's
    per-base probabilities for Human test window chr10:8,388,608-9,437,184
    exactly (all 1,048,576 bases identical).
  - Chimpanzee (`mPanTro3`) runs on single chromosomes (51 Mb and 231 Mb) and
    on two chromosomes took 1.9 s per window on an A40. They peaked at 20 GB
    of GPU memory and 10 GB of RAM. The same chromosome gave identical scores
    in two separate runs.
  - Regions re-derived independently from the bigWig matched the BED file
    exactly.
  - `download_genome.sh` was tested on a small UCSC file.
  - Not tested were the Hugging Face login and download in setup steps 4 and
    5 (the test copied an existing cache of the same revisions), a
    whole-genome run, and GPUs other than the A40.
- **Software.** Exact versions are in `requirements.txt` (Python 3.13.8,
  PyTorch 2.8.0 with CUDA 12.8). Jobs request one A40 GPU, which is the GPU the
  model was trained, scored and tested on.
