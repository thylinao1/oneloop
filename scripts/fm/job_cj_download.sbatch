#!/bin/bash
#SBATCH --job-name=amex-cj-download
#SBATCH --partition=long
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --exclude=xgpj0
#SBATCH --mail-type=FAIL,TIME_LIMIT_90
#SBATCH --mail-user=e1506804@comp.nus.edu.sg
#SBATCH --output=/home/e/e1506804/amex-oneloop/logs/%x-%j.out
#
# Complete Journey data staging: verify the zip, extract, verify row counts + columns,
# convert to parquet, record provenance. PRIMARY PATH: the verified archive is already staged as the
# verified zip at ~/amex-oneloop/cj/source.zip (official dunnhumby CDN asset). FALLBACK:
# set DATA_URL to fetch it (the dunnhumby source-files page is click-through-gated, so
# the fallback exists for a refreshed direct link, not for casual reuse). Either way the
# zip must match the recorded sha256 or nothing proceeds.
# License: dunnhumby terms permit use solely for research, personal or non-commercial
# purposes. Kaggle mirrors are NOT used (no redistribution right).
set -euo pipefail

SCRATCH=/tmp/$SLURM_JOB_ID
mkdir -p "$SCRATCH"/xdg
export XDG_CACHE_HOME=$SCRATCH/xdg
export TMPDIR=$SCRATCH
export PIP_CACHE_DIR=$SCRATCH/xdg/pip
trap 'rm -rf "$SCRATCH"' EXIT

BASE=$HOME/amex-oneloop
CJ=$BASE/cj
FM=$BASE/scripts/fm
mkdir -p "$CJ" "$BASE/logs"
echo "[sbatch] node=$(hostname) job=$SLURM_JOB_ID cpus=$SLURM_CPUS_PER_TASK $(date -u +%FT%TZ)"

EXPECTED_SHA=5e0a3d72fe8562fe0ab995f70fb58b74359e8ec4bbccd1521e2b137da0558f9a
ZIP=$CJ/source.zip
SOURCE_REF="pre-staged $ZIP"

if [ ! -f "$ZIP" ]; then
  if [ -n "${DATA_URL:-}" ]; then
    echo "[sbatch] zip not staged; fetching from DATA_URL"
    # download to a .part file and move only on success, so a partial download can
    # never sit at $ZIP and poison every resubmit
    curl -fSL -o "$ZIP.part" "$DATA_URL"
    mv "$ZIP.part" "$ZIP"
    SOURCE_REF="$DATA_URL"
  else
    echo "FATAL: $ZIP is missing and DATA_URL is not set."
    echo "Stage the verified zip (see CJ-RUNBOOK.md) or export DATA_URL, then resubmit."
    exit 2
  fi
fi

GOT_SHA=$(sha256sum "$ZIP" | awk '{print $1}')
if [ "$GOT_SHA" != "$EXPECTED_SHA" ]; then
  echo "FATAL: source.zip sha256 mismatch"
  echo "  got      $GOT_SHA"
  echo "  expected $EXPECTED_SHA"
  echo "Nothing extracted; nothing trains on an unverified asset."
  exit 3
fi
echo "[sbatch] zip sha256 verified: $GOT_SHA"

# archive layout has spaces in the directory name; keep everything quoted
CSV_DIR="dunnhumby_The-Complete-Journey/dunnhumby_The-Complete-Journey CSV"
EXTRACT=$SCRATCH/extract
mkdir -p "$EXTRACT"
unzip -q -o "$ZIP" -d "$EXTRACT"
if [ ! -d "$EXTRACT/$CSV_DIR" ]; then
  echo "FATAL: expected archive directory not found: $CSV_DIR"
  find "$EXTRACT" -maxdepth 2 -type d
  exit 4
fi
cp "$EXTRACT/$CSV_DIR"/*.csv "$CJ/"
# the official user guide ships inside the archive; keep it next to the data as the
# schema reference the prereg cites
find "$EXTRACT" -maxdepth 3 -name "*.pdf" -exec cp {} "$CJ/" \;
ls -la "$CJ"

echo "[sbatch] building /tmp venv"
python3 -m venv "$SCRATCH/venv"
"$SCRATCH/venv/bin/pip" -q install --no-cache-dir polars numpy
"$SCRATCH/venv/bin/python" -c "import polars, numpy; print('[sbatch] polars', polars.__version__, 'numpy', numpy.__version__)"

"$SCRATCH/venv/bin/python" "$FM/cj_download.py" \
  --dir "$CJ" --source-ref "$SOURCE_REF" --zip-sha256 "$GOT_SHA"

sha256sum "$CJ"/*.csv "$CJ"/*.parquet > "$CJ/CHECKSUMS.txt"
echo "[sbatch] wrote $CJ/CHECKSUMS.txt"
echo "[sbatch] exit OK $(date -u +%FT%TZ)"
