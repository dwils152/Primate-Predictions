#!/bin/bash
# Download job submitted by ./download_genome.sh. Use that script instead of this file.
#SBATCH --job-name=download
#SBATCH --partition=Orion
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -f scripts/common.sh ]] || { echo "Submit this job from the Primate-Predictions folder." >&2; exit 1; }
source scripts/common.sh

name="$1"
url="$2"
folder="genomes/$name"
mkdir -p "$folder"
download="$folder/download.partial"

echo "Downloading $url"
curl --location --fail --retry 3 --retry-delay 30 --output "$download" "$url"
download_sha256=$(sha256sum "$download" | cut -d' ' -f1)

if gzip -t "$download" 2>/dev/null; then
  echo "Decompressing"
  gunzip -c "$download" > "$folder/$name.fa.partial"
  rm "$download"
else
  mv "$download" "$folder/$name.fa.partial"
fi
head -c 1 "$folder/$name.fa.partial" | grep -q '>' ||
  fail "The downloaded file is not a FASTA file (it should start with '>')."
mv "$folder/$name.fa.partial" "$folder/$name.fa"

echo "Indexing"
"$PY" -c 'import pysam, sys; pysam.faidx(sys.argv[1])' "$folder/$name.fa"

"$PY" - "$folder" "$name" "$url" "$download_sha256" <<'EOF'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path

folder, name, url, download_sha256 = sys.argv[1:]
fasta = Path(folder) / f"{name}.fa"
digest = hashlib.sha256()
with fasta.open("rb") as handle:
    for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
        digest.update(chunk)
record = {
    "name": name,
    "url": url,
    "downloaded_utc": datetime.now(timezone.utc).isoformat(),
    "download_sha256": download_sha256,
    "fasta": str(fasta),
    "fasta_sha256": digest.hexdigest(),
}
(Path(folder) / "source.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
EOF

echo
echo "Done. Predict with:   ./predict.sh $name"
