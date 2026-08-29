#!/bin/bash
# Day-1 data fetch; pinned sources per CONTRACT.md §4. Each fetch independent; failures logged, not fatal.
set -u
cd "$(dirname "$0")/.." || exit 1
mkdir -p data
LOG=data/fetch.log; SUMS=data/CHECKSUMS.txt
note() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
sum_it() { [ -f "$1" ] && shasum -a 256 "$1" >> "$SUMS" && note "sha256 recorded: $1 ($(stat -f%z "$1") bytes)"; }

# 1. Criteo Uplift v2.1 (public HF resolve URL, ~311MB)
if [ ! -f data/criteo-uplift-v2.1.csv.gz ]; then
  note "criteo: downloading"
  curl -sSL -o data/criteo-uplift-v2.1.csv.gz \
    "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/criteo-research-uplift-v2.1.csv.gz" \
    && sum_it data/criteo-uplift-v2.1.csv.gz || note "criteo: FAILED"
fi

# 2. Hillstrom (small CSV)
if [ ! -f data/hillstrom.csv ]; then
  note "hillstrom: downloading"
  curl -sSL -o data/hillstrom.csv \
    "http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv" \
    && head -c 200 data/hillstrom.csv | grep -qi "recency" && sum_it data/hillstrom.csv || note "hillstrom: FAILED or unexpected content"
fi

# 3. KNOMAD bilateral 2021 via Wayback (covariates only)
if [ ! -f data/knomad_bilateral_2021.xlsx ]; then
  note "knomad: downloading (wayback)"
  curl -sSL -o data/knomad_bilateral_2021.xlsx \
    "https://web.archive.org/web/20230424090916/https://knomad.org/sites/default/files/2022-12/bilateral_remittance_matrix_2021_0.xlsx" \
    && sum_it data/knomad_bilateral_2021.xlsx || note "knomad: FAILED"
fi

# 4. dunnhumby Complete Journey; zip URL lives in the source-files page source
if [ ! -f data/dunnhumby_complete_journey.zip ]; then
  note "dunnhumby: discovering zip URL from page source"
  DH_URL=$(curl -sSL "https://www.dunnhumby.com/source-files/" | grep -oE 'https://[^"]+[Cc]omplete[-_ ]?[Jj]ourney[^"]*\.zip' | head -1)
  if [ -z "$DH_URL" ]; then
    DH_URL=$(curl -sSL "https://www.dunnhumby.com/source-files/" | grep -oE 'https://[^"]+\.zip' | head -5 | grep -i journey | head -1)
  fi
  if [ -n "$DH_URL" ]; then
    note "dunnhumby: found $DH_URL"
    curl -sSL -o data/dunnhumby_complete_journey.zip "$DH_URL" && sum_it data/dunnhumby_complete_journey.zip || note "dunnhumby: download FAILED"
  else
    note "dunnhumby: URL not found in page source; use Mendeley mirror data.mendeley.com/datasets/7myy93ym6k/1 (manual/agent follow-up)"
  fi
fi

# 5. SingStat monthly International Visitor Arrivals via data.gov.sg
if [ ! -f data/singstat_iva_monthly.csv ]; then
  note "singstat: initiate-download via data.gov.sg API"
  DS=d_7e7b2ee60c6ffc962f80fef129cf306e
  POLL=$(curl -sSL -X GET "https://api-open.data.gov.sg/v1/public/api/datasets/${DS}/poll-download" -H 'Content-Type: application/json')
  URL=$(echo "$POLL" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('data',{}).get('url',''))" 2>/dev/null)
  if [ -z "$URL" ]; then
    curl -sSL -X GET "https://api-open.data.gov.sg/v1/public/api/datasets/${DS}/initiate-download" >/dev/null 2>&1
    sleep 5
    POLL=$(curl -sSL -X GET "https://api-open.data.gov.sg/v1/public/api/datasets/${DS}/poll-download" -H 'Content-Type: application/json')
    URL=$(echo "$POLL" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('data',{}).get('url',''))" 2>/dev/null)
  fi
  if [ -n "$URL" ]; then
    curl -sSL -o data/singstat_iva_monthly.csv "$URL" && sum_it data/singstat_iva_monthly.csv || note "singstat: download FAILED"
  else
    note "singstat: API route failed; agent follow-up: fetch via data.gov.sg UI or SingStat TableBuilder"
  fi
fi

# 6. TabFormer transactions.tgz via Git-LFS batch API (~278MB; quota-flaky; that's why we fetch TODAY)
if [ ! -f data/transactions.tgz ]; then
  note "tabformer: resolving LFS pointer"
  PTR=$(curl -sSL "https://raw.githubusercontent.com/IBM/TabFormer/main/data/credit_card/transactions.tgz")
  OID=$(echo "$PTR" | grep '^oid' | cut -d: -f2)
  SIZE=$(echo "$PTR" | grep '^size' | awk '{print $2}')
  if [ -n "$OID" ] && [ -n "$SIZE" ]; then
    note "tabformer: oid=$OID size=$SIZE"
    HREF=$(curl -sSL -X POST "https://github.com/IBM/TabFormer.git/info/lfs/objects/batch" \
      -H "Accept: application/vnd.git-lfs+json" -H "Content-Type: application/vnd.git-lfs+json" \
      -d "{\"operation\":\"download\",\"transfer\":[\"basic\"],\"objects\":[{\"oid\":\"$OID\",\"size\":$SIZE}]}" \
      | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['objects'][0]['actions']['download']['href'])" 2>/dev/null)
    if [ -n "$HREF" ]; then
      curl -sSL -o data/transactions.tgz "$HREF" && sum_it data/transactions.tgz || note "tabformer: download FAILED"
      ACTUAL=$(stat -f%z data/transactions.tgz 2>/dev/null || echo 0)
      [ "$ACTUAL" = "$SIZE" ] && note "tabformer: size matches pointer ($SIZE)" || note "tabformer: SIZE MISMATCH got=$ACTUAL want=$SIZE"
    else
      note "tabformer: LFS batch API gave no href (quota?); manual fallback: browser-download https://ibm.box.com/v/tabformer-data"
    fi
  else
    note "tabformer: pointer parse FAILED"
  fi
fi

note "fetch pass complete"
ls -la data/ | tee -a "$LOG"