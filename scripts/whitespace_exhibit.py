#!/usr/bin/env python3
"""Whitespace-Head (WS-B3): real-signals SG merchant-signing whitespace ranking,
plus (stage 2) the genuine backbone-to-head wire at CATEGORY level.

Produces results/whitespace.json (CONTRACT §2 schema) + results/whitespace_map_points.json.

Stage 1 design (seed=42, all deterministic):
  1. Filter FSQ OS Places SG slice to plausibly card-accepting categories.
  2. Real observable signals per POI: category MDR-sensitivity prior (cited),
     local density (neighbors within 250m, KD-tree), tourist/premium-zone
     proximity (distance decay to 10 documented zones), chain-vs-independent
     (name frequency >= 3 = chain).
  3. score_real_signals = transparent weighted sum (weights in the JSON).
  4. Closed-loop increment = SENSITIVITY ANALYSIS ONLY: a simulated
     demand-weighted acceptance-gap signal at 3 stated strengths; how the
     top-100 ranking reorders. Labeled 'simulated-increment'.
  5. Pseudonymize: aggregate to grid-cell x category-group buckets with
     area names from an embedded gazetteer. NO real merchant names in output.

Stage 2 (--stage2); the backbone wire, honest by construction:
  The backbone's per-merchant embeddings come from the IBM TabFormer SYNTHETIC
  corpus and share no key space with the real FSQ Singapore merchants, so the
  honest wire is CATEGORY-LEVEL: TabFormer merchant embeddings (pre-cut pooled,
  scripts/fm/embed.py) are aggregated into transaction-weighted MCC-group
  centroids; each FSQ category group maps to a documented MCC set; a bucket's
  score_with_embeddings = cosine similarity (after corpus-mean centering)
  between its category group's centroid and an acceptance-anchor profile built
  from the most card-accepting groups. The RANKING STAYS BY REAL SIGNALS; the
  wire is an added column plus a reported-as-obtained rank-correlation note.
  Label (verbatim in the JSON): category-level wire on synthetic-corpus
  embeddings; at Amex this runs at merchant level on real closed-loop data.

--check: recompute (CPU, deterministic) and compare against the committed JSON
at 1e-6; exit 0 iff numerically identical. If the committed JSON carries a
"wire" block, --check recomputes stage 2 too (auto-enabled).
"""
from __future__ import annotations

import os

# Cap threads BEFORE numeric imports (8GB shared machine; determinism).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

SEED = 42
GRID_DEG = 0.008          # ~890 m cells near the equator
DENSITY_RADIUS_M = 250.0
CHAIN_MIN_COUNT = 3       # name frequency >= 3 in the SG slice => chain
MIN_BUCKET_POIS = 10
MAX_RANKING_ROWS = 400
TOP_N_INLINED = 100
TZ_DECAY_KM = 1.5         # tourist-zone distance-decay length scale
LEAKAGE_STRENGTHS = [0.10, 0.25, 0.50]
M_PER_DEG_LAT = 111_320.0

# Transparent scoring weights (sum to 1.0); stated in the JSON verbatim.
WEIGHTS = {
    "mdr_sensitivity_prior": 0.30,
    "tourist_zone_proximity": 0.30,
    "local_density": 0.25,
    "independent_share": 0.15,
}

# --- Category groups: plausibly card-accepting universe -----------------------
# Group -> (match rules over 'L1' or 'L1 > L2' prefixes of fsq_category_labels)
# Excluded entirely: Community and Government, Landmarks and Outdoors, Event,
# offices, transport infrastructure, hospitals; not merchant-signing targets.
GROUP_RULES = {
    "F&B": [("Dining and Drinking", None)],
    "Retail": [("Retail", None)],
    "Hotels & Lodging": [("Travel and Transportation", "Lodging")],
    "Personal Services": [
        ("Business and Professional Services", "Health and Beauty Service"),
        ("Business and Professional Services", "Automotive Service"),
        ("Business and Professional Services", "Pet Service"),
        ("Travel and Transportation", "Travel Agency"),
    ],
    "Health & Wellness": [
        ("Health and Medicine", "Physician"),
        ("Health and Medicine", "Dentist"),
        ("Health and Medicine", "Healthcare Clinic"),
        ("Health and Medicine", "Alternative Medicine Clinic"),
        ("Health and Medicine", "Acupuncture Clinic"),
        ("Health and Medicine", "Chiropractor"),
        ("Health and Medicine", "Physical Therapy Clinic"),
        ("Health and Medicine", "Optometrist"),
    ],
    "Entertainment & Leisure": [
        ("Arts and Entertainment", "Movie Theater"),
        ("Arts and Entertainment", "Night Club"),
        ("Arts and Entertainment", "Arcade"),
        ("Arts and Entertainment", "Bowling Alley"),
        ("Arts and Entertainment", "Amusement Park"),
        ("Arts and Entertainment", "Gaming Cafe"),
        ("Arts and Entertainment", "Art Gallery"),
        ("Arts and Entertainment", "Museum"),
        ("Sports and Recreation", "Gym and Studio"),
        ("Sports and Recreation", "Golf"),
    ],
}

# Public MDR-sensitivity prior per group (0-1; higher = more likely the 1.5-3%
# card MDR is a rejection driver vs 0.3% QRIS / free PayNow). Values are stated
# modeling priors grounded in cited public facts; citation keys resolve in
# citations.json (Story-owned).
MDR_PRIORS = {
    "F&B": {
        "prior": 0.85,
        "citations": ["mdr_sg_card_range", "sg_small_merchant_mdr_rejection",
                      "qris_mdr_pressure", "fnb_low_margin_cash_preference"],
    },
    "Retail": {
        "prior": 0.65,
        "citations": ["mdr_sg_card_range", "sg_small_merchant_mdr_rejection"],
    },
    "Personal Services": {
        "prior": 0.70,
        "citations": ["mdr_sg_card_range", "qris_mdr_pressure"],
    },
    "Health & Wellness": {
        "prior": 0.45,
        "citations": ["mdr_sg_card_range"],
    },
    "Entertainment & Leisure": {
        "prior": 0.50,
        "citations": ["mdr_sg_card_range"],
    },
    "Hotels & Lodging": {
        "prior": 0.20,
        "citations": ["hotel_card_acceptance_norm"],
    },
}

# --- 10 documented SG tourist/premium zones (approx centroids) ----------------
TOURIST_ZONES = [
    {"name": "Orchard Road", "lat": 1.3048, "lon": 103.8318},
    {"name": "Marina Bay", "lat": 1.2839, "lon": 103.8607},
    {"name": "Sentosa", "lat": 1.2494, "lon": 103.8303},
    {"name": "Changi Jewel", "lat": 1.3602, "lon": 103.9897},
    {"name": "Chinatown", "lat": 1.2838, "lon": 103.8443},
    {"name": "Little India", "lat": 1.3066, "lon": 103.8518},
    {"name": "Bugis", "lat": 1.3009, "lon": 103.8555},
    {"name": "Clarke Quay", "lat": 1.2906, "lon": 103.8465},
    {"name": "Katong / Joo Chiat", "lat": 1.3050, "lon": 103.9020},
    {"name": "Raffles Place", "lat": 1.2841, "lon": 103.8514},
]

# --- Embedded SG area gazetteer (approx centroids) for pseudonymous labels ----
AREA_GAZETTEER = [
    ("Orchard Road", 1.3048, 103.8318), ("Somerset", 1.3006, 103.8390),
    ("Dhoby Ghaut", 1.2993, 103.8455), ("Marina Bay", 1.2839, 103.8607),
    ("Raffles Place", 1.2841, 103.8514), ("Tanjong Pagar", 1.2765, 103.8459),
    ("Maxwell Road", 1.2803, 103.8446), ("Chinatown", 1.2838, 103.8443),
    ("Clarke Quay", 1.2906, 103.8465), ("Boat Quay", 1.2870, 103.8500),
    ("Bugis", 1.3009, 103.8555), ("Little India", 1.3066, 103.8518),
    ("Kampong Glam", 1.3025, 103.8590), ("City Hall", 1.2931, 103.8520),
    ("Esplanade", 1.2897, 103.8555), ("Sentosa", 1.2494, 103.8303),
    ("HarbourFront", 1.2653, 103.8210), ("Tiong Bahru", 1.2862, 103.8320),
    ("Outram", 1.2805, 103.8390), ("River Valley", 1.2937, 103.8360),
    ("Robertson Quay", 1.2907, 103.8390), ("Newton", 1.3120, 103.8380),
    ("Novena", 1.3204, 103.8438), ("Balestier", 1.3250, 103.8480),
    ("Toa Payoh", 1.3343, 103.8474), ("Bishan", 1.3508, 103.8480),
    ("Ang Mo Kio", 1.3691, 103.8454), ("Serangoon", 1.3554, 103.8697),
    ("Hougang", 1.3612, 103.8863), ("Punggol", 1.4041, 103.9025),
    ("Sengkang", 1.3868, 103.8914), ("Tampines", 1.3496, 103.9568),
    ("Pasir Ris", 1.3721, 103.9474), ("Bedok", 1.3236, 103.9273),
    ("Katong", 1.3050, 103.9020), ("Joo Chiat", 1.3122, 103.9010),
    ("East Coast", 1.3010, 103.9120), ("Marine Parade", 1.3020, 103.9070),
    ("Paya Lebar", 1.3177, 103.8930), ("Geylang", 1.3140, 103.8820),
    ("Kallang", 1.3100, 103.8660), ("Lavender", 1.3074, 103.8630),
    ("Farrer Park", 1.3122, 103.8540), ("Changi Airport", 1.3602, 103.9897),
    ("Tanah Merah", 1.3271, 103.9464), ("Simei", 1.3432, 103.9535),
    ("Expo", 1.3350, 103.9614), ("Jurong East", 1.3331, 103.7422),
    ("Jurong West", 1.3404, 103.7090), ("Clementi", 1.3151, 103.7654),
    ("Bukit Timah", 1.3294, 103.8021), ("Holland Village", 1.3110, 103.7961),
    ("Queenstown", 1.2942, 103.7861), ("Buona Vista / one-north", 1.3070, 103.7900),
    ("Bukit Merah", 1.2819, 103.8239), ("Alexandra", 1.2870, 103.8050),
    ("Woodlands", 1.4360, 103.7865), ("Yishun", 1.4294, 103.8350),
    ("Sembawang", 1.4491, 103.8200), ("Bukit Batok", 1.3590, 103.7637),
    ("Bukit Panjang", 1.3774, 103.7719), ("Choa Chu Kang", 1.3840, 103.7470),
    ("Boon Lay / Pioneer", 1.3390, 103.7060), ("Tuas", 1.3200, 103.6500),
    ("Upper Thomson", 1.3540, 103.8320), ("MacPherson / Ubi", 1.3280, 103.8900),
    ("Mount Faber / Telok Blangah", 1.2710, 103.8090),
    ("Seletar", 1.4130, 103.8690), ("Mandai", 1.4100, 103.7890),
]

DATA_SOURCE = {
    "name": "Foursquare OS Places — SG slice (fsq-os-places dt=2026-08-11)",
    "url": "https://huggingface.co/datasets/foursquare/fsq-os-places",
    "sha256": "54effd7b8ead5076090cf65ffa3b2770f006913d854e032536fed8d69c56920e",
}

# --- Stage 2: backbone wire constants ----------------------------------------
WIRE_LABEL = ("category-level wire on synthetic-corpus embeddings; "
              "at Amex this runs at merchant level on real closed-loop data")
EMB_DIM = 512
EMB_CHUNK = 8192          # rows per float64 accumulation chunk (8GB machine)
ANCHOR_TOP_K = 3          # anchor = the K most card-accepting category groups
# sha256 of data/transactions.tgz (same constant as scripts/fm/common.py)
TABFORMER_SHA256 = "e9f589a0958f40d60f81b1a2e8428db86e00c05755caf44fb055827976c0efa2"

# FSQ category group -> MCC codes observed in the TabFormer corpus (modal MCC
# per merchant). Mapping documented verbatim in the JSON "wire" block.
# Deliberate exclusions (documented): 8062 hospitals (the FSQ universe excludes
# hospitals), gambling codes 7995/7801/7802 (no counterpart in the exhibit's
# category universe), wholesale distribution 5045/5094/5192/5193 (not consumer
# storefronts), and all fuel/transit/telecom/utility/money-transfer/airline/
# car-rental/government codes (outside the merchant-signing universe).
# Known approximation (documented): MCC 5813 covers bars AND nightclubs; FSQ
# puts Night Club under Entertainment. 5813 stays in F&B per the FSQ taxonomy's
# own placement of Bars under Dining and Drinking.
MCC_GROUP_CODES = {
    "F&B": {5812, 5813, 5814},
    "Retail": {5211, 5251, 5261, 5300, 5310, 5311, 5411, 5499, 5621, 5651,
               5655, 5661, 5712, 5719, 5722, 5732, 5733, 5912, 5921, 5932,
               5941, 5942, 5947, 5970, 5977},
    "Hotels & Lodging": {7011},   # plus the 3500-3999 lodging-chain range below
    "Personal Services": {4722, 7210, 7230, 7531, 7538, 7542, 7549},
    "Health & Wellness": {8011, 8021, 8041, 8043, 8049, 8099},
    "Entertainment & Leisure": {7832, 7922, 7996},
}
HOTEL_MCC_RANGE = (3500, 3999)    # ISO 18245 lodging-chain block

MCC_NAMES = {
    5812: "eating places and restaurants", 5813: "drinking places (bars, taverns, nightclubs)",
    5814: "fast food restaurants", 5411: "grocery stores and supermarkets",
    5499: "miscellaneous food stores", 5300: "wholesale clubs", 5310: "discount stores",
    5311: "department stores", 5912: "drug stores and pharmacies",
    5921: "package stores: beer, wine, liquor", 5941: "sporting goods stores",
    5942: "book stores", 5947: "gift, card, novelty and souvenir shops",
    5932: "antique shops", 5621: "women's ready-to-wear stores", 5651: "family clothing stores",
    5655: "sports and riding apparel stores", 5661: "shoe stores",
    5712: "furniture and home furnishings stores", 5719: "miscellaneous house furnishing stores",
    5722: "household appliance stores", 5732: "electronics stores", 5733: "music stores",
    5211: "lumber and building materials stores", 5251: "hardware stores",
    5261: "nurseries and lawn and garden supply", 5970: "artist supply and craft shops",
    5977: "cosmetic stores", 7230: "beauty and barber shops", 7210: "laundry and cleaning services",
    7538: "automotive service shops", 7542: "car washes", 7531: "automotive body repair shops",
    7549: "towing services", 4722: "travel agencies and tour operators",
    8011: "doctors and physicians", 8021: "dentists and orthodontists", 8041: "chiropractors",
    8043: "opticians, optical goods and eyeglasses", 8049: "podiatrists and chiropodists",
    8099: "medical services, not elsewhere classified", 7832: "motion picture theaters",
    7996: "amusement parks and carnivals", 7922: "theatrical producers and ticket agencies",
    7011: "hotels, motels and resorts",
}


def mcc_group(mcc: int) -> str | None:
    for g, codes in MCC_GROUP_CODES.items():
        if mcc in codes:
            return g
    if HOTEL_MCC_RANGE[0] <= mcc <= HOTEL_MCC_RANGE[1]:
        return "Hotels & Lodging"
    return None


def find_data(cli_path: str | None) -> Path:
    candidates = []
    if cli_path:
        candidates.append(Path(cli_path))
    here = Path(__file__).resolve()
    candidates.append(here.parent.parent / "data" / "fsq_sg.parquet")
    candidates.append(Path.home() / "Developer" / "amex-ai-hackathon" / "data" / "fsq_sg.parquet")
    for c in candidates:
        if c.is_file():
            return c
    sys.exit(f"ERROR: fsq_sg.parquet not found in {[str(c) for c in candidates]}")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assign_group(labels: list[str] | None) -> str | None:
    """First matching group by GROUP_RULES over any of the POI's labels."""
    if not labels:
        return None
    for group, rules in GROUP_RULES.items():
        for l1, l2 in rules:
            for lab in labels:
                parts = lab.split(" > ")
                if parts[0] != l1:
                    continue
                if l2 is None or (len(parts) > 1 and parts[1] == l2):
                    return group
    return None


def build_poi_frame(data_path: Path) -> pl.DataFrame:
    df = pl.read_parquet(data_path)
    n_universe = df.height
    df = df.filter(
        pl.col("date_closed").is_null()
        & pl.col("latitude").is_between(1.15, 1.48)
        & pl.col("longitude").is_between(103.6, 104.1)
        & pl.col("fsq_category_labels").is_not_null()
        & pl.col("name").is_not_null()
    )
    groups = [assign_group(labels) for labels in df["fsq_category_labels"].to_list()]
    df = df.with_columns(pl.Series("category_group", groups, dtype=pl.Utf8))
    df = df.filter(pl.col("category_group").is_not_null())
    # chain-vs-independent: normalized name frequency across the filtered universe
    df = df.with_columns(
        pl.col("name").str.to_lowercase()
        .str.replace_all(r"[^a-z0-9]+", " ").str.strip_chars()
        .alias("norm_name")
    )
    counts = df.group_by("norm_name").len().rename({"len": "name_count"})
    df = df.join(counts, on="norm_name", how="left")
    df = df.with_columns((pl.col("name_count") >= CHAIN_MIN_COUNT).alias("is_chain"))
    return df, n_universe


def compute_signals(df: pl.DataFrame) -> pl.DataFrame:
    lat = df["latitude"].to_numpy().astype(np.float64)
    lon = df["longitude"].to_numpy().astype(np.float64)
    lat0 = 1.3521
    x = (lon * math.cos(math.radians(lat0)) * M_PER_DEG_LAT).astype(np.float64)
    y = (lat * M_PER_DEG_LAT).astype(np.float64)
    pts = np.column_stack([x, y])

    tree = cKDTree(pts)
    neigh = tree.query_ball_point(pts, r=DENSITY_RADIUS_M, workers=1, return_length=True)
    neigh = np.asarray(neigh, dtype=np.int64) - 1  # exclude self
    p99 = float(np.percentile(neigh, 99))
    dens_norm = np.minimum(np.log1p(neigh) / math.log1p(p99), 1.0).astype(np.float32)

    tz = np.zeros(len(df), dtype=np.float32)
    d_min = np.full(len(df), np.inf, dtype=np.float64)
    nearest_zone = np.zeros(len(df), dtype=np.int32)
    for i, z in enumerate(TOURIST_ZONES):
        zx = z["lon"] * math.cos(math.radians(lat0)) * M_PER_DEG_LAT
        zy = z["lat"] * M_PER_DEG_LAT
        d_km = np.hypot(x - zx, y - zy) / 1000.0
        closer = d_km < d_min
        d_min = np.where(closer, d_km, d_min)
        nearest_zone = np.where(closer, i, nearest_zone)
        tz = np.maximum(tz, np.exp(-d_km / TZ_DECAY_KM).astype(np.float32))

    priors = np.array([MDR_PRIORS[g]["prior"] for g in df["category_group"].to_list()],
                      dtype=np.float32)
    indep = (~df["is_chain"].to_numpy()).astype(np.float32)

    score = (WEIGHTS["mdr_sensitivity_prior"] * priors
             + WEIGHTS["tourist_zone_proximity"] * tz
             + WEIGHTS["local_density"] * dens_norm
             + WEIGHTS["independent_share"] * indep).astype(np.float32)

    return df.with_columns(
        pl.Series("neighbors_250m", neigh),
        pl.Series("density_norm", dens_norm),
        pl.Series("tourist_zone_score", tz),
        pl.Series("nearest_zone_km", d_min.astype(np.float32)),
        pl.Series("nearest_zone_idx", nearest_zone),
        pl.Series("mdr_prior", priors),
        pl.Series("independent", indep),
        pl.Series("poi_score", score),
        (pl.col("latitude") // GRID_DEG).cast(pl.Int32).alias("cell_y"),
        (pl.col("longitude") // GRID_DEG).cast(pl.Int32).alias("cell_x"),
    )


def nearest_area(lat: float, lon: float) -> str:
    best, bd = None, float("inf")
    for name, alat, alon in AREA_GAZETTEER:
        d = (alat - lat) ** 2 + ((alon - lon) * 0.9997) ** 2
        if d < bd:
            bd, best = d, name
    return best


def make_buckets(df: pl.DataFrame) -> list[dict]:
    agg = (
        df.group_by(["cell_x", "cell_y", "category_group"])
        .agg(
            pl.len().alias("n_pois"),
            pl.col("latitude").mean().alias("lat"),
            pl.col("longitude").mean().alias("lon"),
            pl.col("poi_score").mean().alias("score_real_signals"),
            pl.col("tourist_zone_score").mean().alias("tz_mean"),
            pl.col("density_norm").mean().alias("dens_mean"),
            pl.col("neighbors_250m").median().alias("neigh_median"),
            pl.col("nearest_zone_km").mean().alias("zone_km_mean"),
            pl.col("mdr_prior").mean().alias("mdr_prior_mean"),
            pl.col("independent").mean().alias("indep_share"),
        )
        .filter(pl.col("n_pois") >= MIN_BUCKET_POIS)
        .sort(["cell_x", "cell_y", "category_group"])  # stable order pre-RNG
    )
    buckets = agg.to_dicts()
    for b in buckets:
        b["area"] = nearest_area(b["lat"], b["lon"])
        # deterministic nearest zone from bucket centroid (ties broken by list order)
        b["zone_idx"] = min(
            range(len(TOURIST_ZONES)),
            key=lambda i: ((TOURIST_ZONES[i]["lat"] - b["lat"]) ** 2
                           + ((TOURIST_ZONES[i]["lon"] - b["lon"]) * 0.9997) ** 2),
        )
        b["bucket_label"] = f"{b['category_group']} cluster — {b['area']} area"
    # Disambiguate duplicate labels (several grid cells in one area x group):
    # best-scoring cell keeps the plain label; others get "· sector k" (score order).
    by_label: dict[str, list[int]] = {}
    for i, b in enumerate(buckets):
        by_label.setdefault(b["bucket_label"], []).append(i)
    for label, idxs in by_label.items():
        if len(idxs) > 1:
            idxs.sort(key=lambda i: (-buckets[i]["score_real_signals"],
                                     buckets[i]["cell_x"], buckets[i]["cell_y"]))
            for k, i in enumerate(idxs[1:], start=2):
                buckets[i]["bucket_label"] = f"{label} · sector {k}"
    return buckets


def reasons_for(b: dict) -> list[str]:
    zone = TOURIST_ZONES[int(b["zone_idx"])]["name"]
    comps = [
        (WEIGHTS["mdr_sensitivity_prior"] * b["mdr_prior_mean"],
         f"MDR-sensitive category mix ({b['category_group']}, prior {b['mdr_prior_mean']:.2f}: "
         f"1.5-3% card MDR vs 0.3% QRIS anchor)"),
        (WEIGHTS["tourist_zone_proximity"] * b["tz_mean"],
         f"Premium-demand corridor ({b['zone_km_mean']:.1f} km to {zone}; "
         f"zone score {b['tz_mean']:.2f})"),
        (WEIGHTS["local_density"] * b["dens_mean"],
         f"Dense commercial cluster (median {int(b['neigh_median'])} POIs within 250 m)"),
        (WEIGHTS["independent_share"] * b["indep_share"],
         f"{100 * b['indep_share']:.0f}% independent (non-chain) merchants"),
    ]
    comps.sort(key=lambda t: -t[0])
    return [c[1] for c in comps[:3]]


def sensitivity(buckets: list[dict]) -> dict:
    """Simulated demand-weighted acceptance-gap signal at 3 strengths.

    gap_b ~ clip(Normal(mu=mdr_prior_mean_b, sigma=0.15), 0, 1)   [seeded]
    demand_b = 0.6*tz_mean + 0.4*dens_mean
    signal_b = z-score(gap_b * demand_b)
    score(lambda) = (1-lambda)*z(score_real) + lambda*signal_b
    """
    rng = np.random.default_rng(SEED)
    n = len(buckets)
    mu = np.array([b["mdr_prior_mean"] for b in buckets], dtype=np.float64)
    gap = np.clip(rng.normal(mu, 0.15), 0.0, 1.0)
    demand = np.array([0.6 * b["tz_mean"] + 0.4 * b["dens_mean"] for b in buckets])
    signal = gap * demand
    z_sig = (signal - signal.mean()) / signal.std()
    base = np.array([b["score_real_signals"] for b in buckets], dtype=np.float64)
    z_base = (base - base.mean()) / base.std()

    order0 = np.argsort(-base, kind="stable")
    rank0 = np.empty(n, dtype=np.int64)
    rank0[order0] = np.arange(1, n + 1)

    summary = {"strengths": LEAKAGE_STRENGTHS, "spearman_full": {}, "spearman_top100": {},
               "example_movers": {}}
    per_bucket = [[] for _ in range(n)]
    top100 = order0[:min(100, n)]
    for lam in LEAKAGE_STRENGTHS:
        s = (1 - lam) * z_base + lam * z_sig
        order = np.argsort(-s, kind="stable")
        rank = np.empty(n, dtype=np.int64)
        rank[order] = np.arange(1, n + 1)
        key = f"{lam:.2f}"
        rho_full = float(spearmanr(rank0, rank).statistic)
        rho_top = float(spearmanr(rank0[top100], rank[top100]).statistic)
        summary["spearman_full"][key] = round(rho_full, 4)
        summary["spearman_top100"][key] = round(rho_top, 4)
        moves = rank0[top100].astype(int) - rank[top100].astype(int)  # + = climbs
        idx_sorted = np.argsort(-np.abs(moves), kind="stable")[:3]
        summary["example_movers"][key] = [
            {"bucket_label": buckets[int(top100[j])]["bucket_label"],
             "rank_base": int(rank0[top100[j]]),
             "rank_with_signal": int(rank[top100[j]]),
             "moved": int(moves[j])}
            for j in idx_sorted
        ]
        for i in range(n):
            per_bucket[i].append({"leakage_strength": lam, "rank": int(rank[i])})
    return {"summary": summary, "per_bucket": per_bucket, "rank_base": rank0}


# ------------------------------------------------------------ stage 2: wire --

def derive_merchant_mcc(tgz_path: Path, cache_path: Path) -> pl.DataFrame:
    """(merchant_name, merchant_city) -> modal MCC from the raw TabFormer CSV.

    Streams the tgz member to a temp file (deleted after), aggregates with a
    lazy 3-column projection, takes the modal MCC per merchant (ties broken by
    lowest code). Keys match scripts/fm/prep.py: raw strings, nulls -> "NA".
    """
    import shutil
    import tarfile

    got = sha256_file(tgz_path)
    if got != TABFORMER_SHA256:
        sys.exit(f"ERROR: sha256 mismatch for {tgz_path}: {got}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_csv = cache_path.parent / "_tabformer_tmp.csv"
    print(f"[stage2] deriving merchant->MCC mapping from {tgz_path} (one-time; cached after)")
    try:
        with tarfile.open(tgz_path, "r:gz") as tf:
            member = next(m for m in tf.getmembers() if m.name.endswith(".csv"))
            src = tf.extractfile(member)
            with open(tmp_csv, "wb") as out:
                shutil.copyfileobj(src, out, length=1 << 20)
        agg = (
            pl.scan_csv(tmp_csv, schema_overrides={
                "Merchant Name": pl.Utf8, "Merchant City": pl.Utf8, "MCC": pl.Int64})
            .select(
                pl.col("Merchant Name").fill_null("NA").alias("merchant_name"),
                pl.col("Merchant City").fill_null("NA").alias("merchant_city"),
                pl.col("MCC").alias("mcc"),
            )
            .group_by(["merchant_name", "merchant_city", "mcc"]).len()
            .collect(engine="streaming")
        )
    finally:
        tmp_csv.unlink(missing_ok=True)
    modal = (
        agg.sort(["merchant_name", "merchant_city", "len", "mcc"],
                 descending=[False, False, True, False])
        .group_by(["merchant_name", "merchant_city"], maintain_order=True)
        .agg(pl.col("mcc").first(), pl.col("len").sum().alias("n_txns"),
             pl.len().alias("n_mcc_distinct"))
        .sort(["merchant_name", "merchant_city"])
    )
    modal.write_parquet(cache_path)
    return modal


def mapping_content_sha256(mapping: pl.DataFrame) -> str:
    """Deterministic content hash of the merchant->MCC mapping (tamper check
    for the gitignored cache; recorded in the JSON, recomputed on --check)."""
    h = hashlib.sha256()
    m = mapping.sort(["merchant_name", "merchant_city"])
    for name, city, mcc, n_txns in zip(
            m["merchant_name"].to_list(), m["merchant_city"].to_list(),
            m["mcc"].to_list(), m["n_txns"].to_list()):
        h.update(f"{name}\x1f{city}\x1f{mcc}\x1f{n_txns}\n".encode())
    return h.hexdigest()


def load_mcc_mapping(tgz_path: Path, cache_path: Path) -> pl.DataFrame:
    if cache_path.is_file():
        try:
            m = pl.read_parquet(cache_path)
            if {"merchant_name", "merchant_city", "mcc", "n_txns", "n_mcc_distinct"} <= set(m.columns):
                return m.sort(["merchant_name", "merchant_city"])
        except Exception as e:  # corrupt cache -> rebuild
            print(f"[stage2] cache unreadable ({e}); rebuilding")
    return derive_merchant_mcc(tgz_path, cache_path)


def compute_wire(emb_path: Path, mapping: pl.DataFrame) -> tuple[dict, dict[str, float]]:
    """Category-level backbone wire. Returns (wire block sans reorder, per-group score).

    Deterministic: fixed group order, rows sorted by (merchant_name,
    merchant_city), chunked float64 accumulation, einsum dot products.
    """
    emb_sha = sha256_file(emb_path)
    emb_cols = [f"emb_{i}" for i in range(EMB_DIM)]
    keys = (pl.scan_parquet(emb_path)
            .select("merchant_name", "merchant_city", "n_txns_pre_cut")
            .collect())
    n_embedded = keys.height
    j = keys.join(mapping.select("merchant_name", "merchant_city", "mcc"),
                  on=["merchant_name", "merchant_city"], how="left")
    n_unjoined = j["mcc"].null_count()
    j = j.with_columns(pl.Series(
        "category_group",
        [mcc_group(m) if m is not None else None for m in j["mcc"].to_list()],
        dtype=pl.Utf8))
    mapped = j.filter(pl.col("category_group").is_not_null())
    groups = sorted(GROUP_RULES)  # the exhibit's own 6 category groups

    sums: dict[str, np.ndarray] = {}
    wts: dict[str, float] = {}
    stats: dict[str, dict] = {}
    for g in groups:
        sel = (mapped.filter(pl.col("category_group") == g)
               .select("merchant_name", "merchant_city"))
        sub = (pl.scan_parquet(emb_path)
               .join(sel.lazy(), on=["merchant_name", "merchant_city"], how="inner")
               .sort(["merchant_name", "merchant_city"])
               .select(emb_cols + ["n_txns_pre_cut"])
               .collect(engine="streaming"))
        M = sub.select(emb_cols).to_numpy()  # float32 [n, 512]
        w = sub["n_txns_pre_cut"].to_numpy().astype(np.float64)
        s = np.zeros(EMB_DIM, dtype=np.float64)
        for c0 in range(0, len(w), EMB_CHUNK):
            s += (M[c0:c0 + EMB_CHUNK].astype(np.float64)
                  * w[c0:c0 + EMB_CHUNK, None]).sum(axis=0)
        sums[g], wts[g] = s, float(w.sum())
        stats[g] = {"n_merchants_embedded": int(len(w)), "n_txns_pre_cut": int(w.sum())}
        print(f"[stage2] centroid {g}: {len(w)} merchants, {int(w.sum())} pre-cut txns")

    total_w = sum(wts.values())
    mu = sum(sums.values()) / total_w  # corpus-wide txn-weighted mean (centering)
    cent = {g: sums[g] / wts[g] for g in groups}

    priors = {g: MDR_PRIORS[g]["prior"] for g in groups}
    anchor_groups = sorted(groups, key=lambda g: (priors[g], g))[:ANCHOR_TOP_K]
    aw = {g: 1.0 - priors[g] for g in anchor_groups}
    z = sum(aw.values())
    anchor_raw = sum((aw[g] / z) * cent[g] for g in anchor_groups)
    anchor_cen = sum((aw[g] / z) * (cent[g] - mu) for g in anchor_groups)

    def cos(a: np.ndarray, b: np.ndarray) -> float:
        na = math.sqrt(float(np.einsum("i,i->", a, a)))
        nb = math.sqrt(float(np.einsum("i,i->", b, b)))
        return float(np.einsum("i,i->", a, b)) / (na * nb)

    scores = {g: round(cos(cent[g] - mu, anchor_cen), 6) for g in groups}
    similarity = sorted(
        ({"group": g,
          "cos_to_anchor": scores[g],
          "cos_to_anchor_raw": round(cos(cent[g], anchor_raw), 6),
          "in_anchor": g in anchor_groups,
          **stats[g]} for g in groups),
        key=lambda r: -r["cos_to_anchor"])

    observed = (mapped.group_by("category_group")
                .agg(pl.col("mcc").unique().sort().alias("codes")))
    codes_by_group = dict(zip(observed["category_group"].to_list(),
                              observed["codes"].to_list()))
    mcc_mapping = {
        g: {"mcc_codes": [int(c) for c in codes_by_group.get(g, [])],
            **stats[g]}
        for g in groups}
    n_codes_mapped = sum(len(v["mcc_codes"]) for v in mcc_mapping.values())

    wire = {
        "label": WIRE_LABEL,
        "status_note": ("score_with_embeddings varies at category-group level "
                        "(six values across the ranking), by construction of the category-level wire"),
        "method": {
            "summary": ("TabFormer per-merchant backbone embeddings (pre-cut pooled, scripts/fm/embed.py) "
                        "are aggregated into transaction-weighted MCC-group centroids; each FSQ category "
                        "group maps to a documented MCC set; a bucket's score_with_embeddings is the cosine "
                        "similarity between its category group's centroid and an acceptance-anchor profile "
                        "built from the most card-accepting groups"),
            "centroid": ("transaction-weighted mean of merchant embeddings (weight = n_txns_pre_cut), "
                         "equal to the mean over that MCC group's pre-cut transaction encodings"),
            "centering": ("cosines computed after subtracting the corpus-wide transaction-weighted mean "
                          "embedding (anisotropy correction for pooled transformer embeddings); raw cosines "
                          "reported alongside"),
            "anchor": ("acceptance-weighted mean of the centroids of the most card-accepting category "
                       "groups; acceptance weight = 1 minus the exhibit's own cited MDR-sensitivity prior"),
            "anchor_groups": {g: round(1.0 - priors[g], 2) for g in anchor_groups},
            "self_inclusion_note": ("groups inside the anchor score high partly by construction; the "
                                    "informative readout is the ordering of the non-anchor, MDR-sensitive groups"),
            "ranking_note": ("the page ranking stays by real signals; this column is an added signal, "
                             "not a re-ranking"),
            "taxonomy_notes": [
                "MCC 5813 covers bars and nightclubs; FSQ places Night Club under Entertainment; "
                "5813 stays in F&B per FSQ's own placement of Bars under Dining and Drinking",
                "hospitals (8062), gambling (7995, 7801, 7802) and wholesale distribution "
                "(5045, 5094, 5192, 5193) are deliberately unmapped: no counterpart in the "
                "exhibit's card-accepting category universe",
                "3500-3999 is the ISO 18245 lodging-chain block; observed codes map to Hotels & Lodging",
            ],
        },
        "mcc_mapping": mcc_mapping,
        "mcc_code_names": {str(k): v for k, v in sorted(MCC_NAMES.items())},
        "mapping_stats": {
            "n_merchants_tabformer": int(mapping.height),
            "n_multi_mcc_merchants": int(mapping.filter(pl.col("n_mcc_distinct") > 1).height),
            "n_embedded_merchants": n_embedded,
            "n_embedded_unjoined": int(n_unjoined),
            "n_embedded_mapped": int(mapped.height),
            "n_mcc_codes_mapped": n_codes_mapped,
            "mapping_rule": "modal MCC per (merchant_name, merchant_city); ties broken by lowest code",
            "mapping_content_sha256": mapping_content_sha256(mapping),
            "cache": "results/cache/tabformer_merchant_mcc.parquet (gitignored; --check rebuilds it from data/transactions.tgz when absent)",
        },
        "embedding_source": {
            "file": "data/merchant_embeddings.parquet",
            "sha256": emb_sha,
            "n_merchants": n_embedded,
            "dim": EMB_DIM,
            "produced_by": "scripts/fm/embed.py (pretrained backbone, pre-cut pooled; cluster job 748987)",
        },
        "group_similarity": similarity,
    }
    return wire, scores


def build(data_path: Path, stage2: bool = False,
          emb_path: Path | None = None, tgz_path: Path | None = None,
          cache_path: Path | None = None) -> tuple[dict, dict]:
    df, n_universe = build_poi_frame(data_path)
    df = compute_signals(df)
    buckets = make_buckets(df)
    sens = sensitivity(buckets)
    rank0 = sens["rank_base"]

    wire, emb_scores = None, None
    if stage2:
        mapping = load_mcc_mapping(tgz_path, cache_path)
        wire, emb_scores = compute_wire(emb_path, mapping)

    order = np.argsort(rank0, kind="stable")
    ranking = []
    for pos in order[:MAX_RANKING_ROWS]:
        b = buckets[int(pos)]
        ranking.append({
            "rank": int(rank0[int(pos)]),
            "bucket_label": b["bucket_label"],
            "area": b["area"],
            "category": b["category_group"],
            "n_pois": int(b["n_pois"]),
            "centroid": {"lat": round(float(b["lat"]), 5), "lon": round(float(b["lon"]), 5)},
            "score_real_signals": round(float(b["score_real_signals"]), 6),
            "score_with_embeddings": (emb_scores[b["category_group"]]
                                      if emb_scores is not None else None),
            "signals": {
                "mdr_sensitivity_prior": round(float(b["mdr_prior_mean"]), 4),
                "tourist_zone_score": round(float(b["tz_mean"]), 4),
                "density_norm": round(float(b["dens_mean"]), 4),
                "independent_share": round(float(b["indep_share"]), 4),
            },
            "sensitivity": sens["per_bucket"][int(pos)],
            "reasons": reasons_for(b),
        })

    if wire is not None:
        # rank-correlation between the real-signals ordering and the
        # embedding-similarity ordering, reported AS OBTAINED (six category-level
        # values produce heavy ties; spearmanr assigns average ranks).
        real_ranked = [r["score_real_signals"] for r in ranking]
        emb_ranked = [r["score_with_embeddings"] for r in ranking]
        rho_ranked = float(spearmanr(real_ranked, emb_ranked).statistic)
        real_all = [b["score_real_signals"] for b in buckets]
        emb_all = [emb_scores[b["category_group"]] for b in buckets]
        rho_all = float(spearmanr(real_all, emb_all).statistic)
        wire["rank_reorder"] = {
            "spearman_ranked": round(rho_ranked, 4),
            "n_ranked": len(ranking),
            "spearman_all_buckets": round(rho_all, 4),
            "n_all_buckets": len(buckets),
            "note": ("as obtained; the page ranking stays by real signals. A value near zero "
                     "means the embedding-similarity ordering is close to independent of the "
                     "real-signals ranking, so blending it in would reorder substantially. "
                     "Ties from six category-level values are handled by average ranks."),
        }

    import scipy, sklearn  # noqa: F401; versions for the envelope

    versions = {
        "python": ".".join(map(str, sys.version_info[:3])),
        "polars": pl.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    data_sources = [DATA_SOURCE]
    labels = ["real-signals base", "simulated-increment sensitivity", "pseudonymized"]
    if wire is not None:
        data_sources = data_sources + [
            {"name": "IBM TabFormer credit-card transactions (SYNTHETIC corpus; wire stage only)",
             "url": "https://github.com/IBM/TabFormer (data/credit_card/transactions.tgz)",
             "sha256": TABFORMER_SHA256},
            {"name": "One Loop backbone per-merchant embeddings (TabFormer, pre-cut pooled)",
             "url": "generated by scripts/fm/embed.py -> data/merchant_embeddings.parquet",
             "sha256": wire["embedding_source"]["sha256"]},
        ]
        labels = labels + ["synthetic-corpus embeddings (category-level wire)"]

    result = {
        "seed": SEED,
        "versions": versions,
        "generated_by": "scripts/whitespace_exhibit.py --check-able",
        "data_sources": data_sources,
        "labels": labels,
        "universe": "FSQ OS Places SG slice (gated HF)",
        "n_pois": n_universe,
        "n_pois_card_accepting_universe": int(df.height),
        "n_buckets": len(buckets),
        "pseudonymized": True,
        "topN_inlined": TOP_N_INLINED,
        "embeddings_status": (WIRE_LABEL if wire is not None
                              else "pending stage 2 (amex-backbone job)"),
        "method": {
            "bucketing": f"~{GRID_DEG:.3f}-deg grid cell x category-group; "
                         f"buckets with >= {MIN_BUCKET_POIS} POIs; top {MAX_RANKING_ROWS} ranked",
            "score_real_signals": "weighted sum of 4 real observable signals (bucket = mean of POI scores)",
            "signal_weights": WEIGHTS,
            "density": f"POI count within {int(DENSITY_RADIUS_M)} m (cKDTree), log-normalized at p99 cap",
            "tourist_zones": TOURIST_ZONES,
            "tourist_zone_decay_km": TZ_DECAY_KM,
            "chain_rule": f"normalized name frequency >= {CHAIN_MIN_COUNT} in SG slice = chain",
            "mdr_priors": MDR_PRIORS,
            "citations_note": "prior values are stated modeling priors; cited facts resolve via citations.json keys",
        },
        "simulation_params": {
            "label": "simulated-increment",
            "what": "demand-weighted acceptance-gap signal (SIMULATED — no acceptance data observed)",
            "gap_model": "gap_b ~ clip(Normal(mu=bucket mean MDR prior, sigma=0.15), 0, 1), rng seed 42",
            "demand_model": "demand_b = 0.6*tourist_zone_score + 0.4*density_norm (bucket means)",
            "blend": "score(lambda) = (1-lambda)*z(score_real) + lambda*z(gap*demand)",
            "leakage_strengths": LEAKAGE_STRENGTHS,
        },
        "sensitivity_summary": sens["summary"],
        "ranking": ranking,
    }
    if wire is not None:
        result["wire"] = wire

    map_points = {
        "seed": SEED,
        "versions": versions,
        "generated_by": "scripts/whitespace_exhibit.py --check-able",
        "data_sources": [DATA_SOURCE],
        "labels": ["real-signals base", "pseudonymized"],
        "points": [
            {"bucket_label": r["bucket_label"], "area": r["area"], "category": r["category"],
             "lat": r["centroid"]["lat"], "lon": r["centroid"]["lon"],
             "score_real_signals": r["score_real_signals"], "n_pois": r["n_pois"],
             "rank": r["rank"]}
            for r in ranking[:MAX_RANKING_ROWS]
        ],
    }
    return result, map_points


def compare_numeric(a, b, path="$", tol=1e-6) -> list[str]:
    diffs = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in ("versions",):
                continue
            if k not in a or k not in b:
                diffs.append(f"{path}.{k}: missing on one side")
            else:
                diffs.extend(compare_numeric(a[k], b[k], f"{path}.{k}", tol))
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append(f"{path}: length {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            diffs.extend(compare_numeric(x, y, f"{path}[{i}]", tol))
    elif isinstance(a, bool) or isinstance(b, bool):
        if a != b:
            diffs.append(f"{path}: {a} != {b}")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if not math.isclose(float(a), float(b), rel_tol=0, abs_tol=tol):
            diffs.append(f"{path}: {a} != {b}")
    elif a != b:
        diffs.append(f"{path}: {a!r} != {b!r}")
    return diffs[:20]


def main():
    repo = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="path to fsq_sg.parquet")
    ap.add_argument("--out", default=None, help="results dir (default <repo>/results)")
    ap.add_argument("--stage2", action="store_true",
                    help="compute the category-level backbone wire (score_with_embeddings + wire block)")
    ap.add_argument("--emb", default=str(repo / "data" / "merchant_embeddings.parquet"),
                    help="backbone per-merchant embeddings parquet (stage 2)")
    ap.add_argument("--tgz", default=str(repo / "data" / "transactions.tgz"),
                    help="TabFormer raw tgz, used only if the MCC mapping cache is absent (stage 2)")
    ap.add_argument("--cache", default=str(repo / "results" / "cache" / "tabformer_merchant_mcc.parquet"),
                    help="merchant->MCC mapping cache (gitignored; rebuilt from --tgz when absent)")
    ap.add_argument("--check", action="store_true",
                    help="recompute deterministically and compare to committed JSON at 1e-6 "
                         "(stage 2 auto-enabled when the committed JSON carries a wire block)")
    args = ap.parse_args()

    data_path = find_data(args.data)
    got = sha256_file(data_path)
    if got != DATA_SOURCE["sha256"]:
        sys.exit(f"ERROR: sha256 mismatch for {data_path}: {got}")

    out_dir = Path(args.out) if args.out else repo / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    ws_path = out_dir / "whitespace.json"
    map_path = out_dir / "whitespace_map_points.json"

    stage2 = args.stage2
    saved = None
    if args.check:
        if not ws_path.is_file():
            sys.exit(f"--check: {ws_path} does not exist")
        saved = json.loads(ws_path.read_text())
        if "wire" in saved and not stage2:
            stage2 = True
            print("--check: committed JSON carries a wire block; stage 2 auto-enabled")

    emb_path = Path(args.emb)
    if stage2 and not emb_path.is_file():
        sys.exit(f"ERROR: --stage2 needs {emb_path} (backbone merchant embeddings)")

    result, map_points = build(data_path, stage2=stage2, emb_path=emb_path,
                               tgz_path=Path(args.tgz), cache_path=Path(args.cache))

    if args.check:
        diffs = compare_numeric(saved, result)
        if diffs:
            print("--check FAILED:", *diffs, sep="\n  ")
            sys.exit(1)
        print(f"--check OK: {ws_path} reproduced numerically (1e-6), "
              f"{len(result['ranking'])} ranked buckets"
              + (", wire block included" if stage2 else ""))
        return

    ws_path.write_text(json.dumps(result, indent=1) + "\n")
    map_path.write_text(json.dumps(map_points, indent=1) + "\n")
    print(f"wrote {ws_path} ({ws_path.stat().st_size:,} B, {len(result['ranking'])} ranked buckets "
          f"of {result['n_buckets']} formed) and {map_path} ({len(map_points['points'])} points)")
    top = result["ranking"][:5]
    for r in top:
        print(f"  #{r['rank']:>3} {r['score_real_signals']:.4f} {r['bucket_label']} (n={r['n_pois']})")
    print("sensitivity spearman_top100:", result["sensitivity_summary"]["spearman_top100"])
    if stage2:
        w = result["wire"]
        print("wire group_similarity (centered cosine to accepting anchor):")
        for r in w["group_similarity"]:
            tag = " [anchor]" if r["in_anchor"] else ""
            print(f"  {r['cos_to_anchor']:+.4f} {r['group']}{tag} ({r['n_merchants_embedded']} merchants)")
        rr = w["rank_reorder"]
        print(f"wire rank_reorder: spearman_ranked={rr['spearman_ranked']} (n={rr['n_ranked']}), "
              f"spearman_all_buckets={rr['spearman_all_buckets']} (n={rr['n_all_buckets']})")


if __name__ == "__main__":
    main()
