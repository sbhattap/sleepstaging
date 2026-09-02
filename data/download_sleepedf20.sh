#!/usr/bin/env bash
# Download the 39 Sleep-Cassette recordings of Sleep-EDF-20 (subjects 0-19, PSG + hypnogram) from
# PhysioNet's current URL and verify every file against PhysioNet's SHA256SUMS.txt.
#
# Our version of the AttnSleep download script
# (https://gist.github.com/emadeldeen24/a22691e36759934e53984289a94cb09b), which uses the retired
# physiobank path. Hypnogram names do not follow the PSG names (SC4001E0-PSG.edf goes with
# SC4001EC-Hypnogram.edf, SC4011E0-PSG.edf with SC4011EH-Hypnogram.edf), so file names are read from
# SHA256SUMS.txt instead of being guessed.
#
# Size: 78 files, about 1.96 GB in total (largest PSG 52 MB). Files are written to
# <out_dir>/SC4<ss><n>/, the layout that `uv run python -m src.prepare_data` reads. prepare_data deletes
# each raw recording once it has been processed.
#
# Usage:
#   bash data/download_sleepedf20.sh                 # all 20 subjects into data/raw
#   SUBJECTS="0 1" bash data/download_sleepedf20.sh  # a subset (e.g. a quick test)
#   bash data/download_sleepedf20.sh /path/to/raw    # another output folder
set -euo pipefail

BASE="https://physionet.org/files/sleep-edfx/1.0.0"   # PhysioNet redirects files to S3, hence curl -L
OUT="${1:-data/raw}"
SUBJECTS="${SUBJECTS:-$(seq 0 19)}"
SUMS="$OUT/meta/SHA256SUMS.txt"

mkdir -p "$OUT/meta"
[ -s "$SUMS" ] || curl -fsSL --retry 3 -o "$SUMS" "$BASE/SHA256SUMS.txt"

n_rec=0
for s in $SUBJECTS; do
  for night in 1 2; do
    stem=$(printf "SC4%02d%d" "$s" "$night")
    # Lines look like "<sha256> sleep-cassette/SC4001E0-PSG.edf"
    lines=$(grep -E " sleep-cassette/${stem}[A-Z0-9]{2}-(PSG|Hypnogram)\.edf$" "$SUMS" || true)
    if [ -z "$lines" ]; then
      echo "skip $stem: not in the release (subject 13 has only night 1)"
      continue
    fi
    # Fail loudly if PhysioNet ever lists more or fewer than one PSG and one hypnogram per night.
    [ "$(grep -c -- '-PSG.edf$' <<<"$lines")" -eq 1 ] && [ "$(grep -c -- '-Hypnogram.edf$' <<<"$lines")" -eq 1 ] \
      || { echo "unexpected file set for $stem:"; echo "$lines"; exit 1; }
    mkdir -p "$OUT/$stem"
    while read -r sha path; do
      f="$OUT/$stem/${path#sleep-cassette/}"
      if [ -f "$f" ] && echo "$sha  $f" | sha256sum -c --quiet - 2>/dev/null; then
        continue  # already downloaded and verified
      fi
      curl -fsSL --retry 3 -o "$f.part" "$BASE/$path"
      echo "$sha  $f.part" | sha256sum -c --quiet - || { echo "checksum mismatch: $path"; rm -f "$f.part"; exit 1; }
      mv "$f.part" "$f"
    done <<<"$lines"
    n_rec=$((n_rec + 1))
    echo "ok $stem"
  done
done

echo "$n_rec recordings downloaded and verified in $OUT ($(du -sh "$OUT" | cut -f1))"
if [ "$SUBJECTS" = "$(seq 0 19)" ] && [ "$n_rec" -ne 39 ]; then
  echo "expected 39 recordings for Sleep-EDF-20, got $n_rec"; exit 1
fi
