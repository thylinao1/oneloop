"""atlas_build.py: merchant embedding atlas (the committed atlas.json envelope).

Loads the merchant embeddings the main backbone job produced, projects them to
2D (UMAP if importable in the job env, else PCA; 'method' records which),
buckets merchants by coarse MCC group (modal MCC per Merchant Name+City,
recomputed from the raw CSV with the same key construction as fm/prep.py), and
samples <=3,000 points pseudonymously (coordinates + mcc_group + size only ; 
no merchant names, ids, or locations in the output).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from common import TABFORMER_SHA256, TABFORMER_URL, atomic_write_json, versions_dict

MCC_GROUPS = [
    (1500, "Agricultural"),
    (3000, "Contracted Services"),
    (3300, "Airlines"),
    (3500, "Car Rental"),
    (4000, "Lodging"),
    (4800, "Transportation"),
    (5000, "Utilities & Telecom"),
    (5600, "Retail Goods"),
    (5700, "Apparel"),
    (6000, "Misc Retail & Dining"),
    (7000, "Financial Services"),
    (7300, "Personal Services"),
    (8000, "Business & Entertainment"),
    (9000, "Professional & Membership"),
    (10000, "Government"),
]


def mcc_group(mcc: int | None) -> str:
    if mcc is None:
        return "Unknown"
    for hi, name in MCC_GROUPS:
        if mcc < hi:
            return name
    return "Unknown"


def modal_mcc_per_merchant(csv_path: str) -> pl.DataFrame:
    tx = pl.read_csv(
        csv_path,
        columns=["Merchant Name", "Merchant City", "MCC"],
        schema_overrides={"Merchant Name": pl.Utf8, "Merchant City": pl.Utf8, "MCC": pl.Int64},
    ).with_columns(
        pl.col("Merchant Name").fill_null("NA"),
        pl.col("Merchant City").fill_null("NA"),  # same fill as fm/prep.py merch_key
    )
    return tx.group_by(["Merchant Name", "Merchant City"]).agg(
        pl.col("MCC").drop_nulls().mode().first().alias("mcc")
    ).rename({"Merchant Name": "merchant_name", "Merchant City": "merchant_city"})


def project_2d(X: np.ndarray, seed: int):
    """Return (xy [n,2], method, method_detail). UMAP if available, else PCA."""
    from sklearn.decomposition import PCA

    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-8
    Xs = (X - mu) / sd
    n_pre = min(50, Xs.shape[1])
    pca = PCA(n_components=n_pre, random_state=seed)
    Xp = pca.fit_transform(Xs)
    try:
        import umap  # optional: pip-installed into the job's /tmp target

        xy = umap.UMAP(
            n_components=2, n_neighbors=15, min_dist=0.1, random_state=seed, verbose=True
        ).fit_transform(Xp)
        detail = {
            "umap": getattr(umap, "__version__", "?"),
            "pipeline": f"standardize -> PCA({n_pre}) -> UMAP(n_neighbors=15, min_dist=0.1)",
        }
        return np.asarray(xy), "UMAP", detail
    except Exception as e:  # numba/numpy mismatch etc. -> honest PCA fallback
        print(f"[atlas] UMAP unavailable ({type(e).__name__}: {e}) -> PCA fallback", flush=True)
        detail = {
            "pipeline": f"standardize -> PCA(2 of {n_pre})",
            "explained_variance_ratio_2d": [float(v) for v in pca.explained_variance_ratio_[:2]],
        }
        return Xp[:, :2], "PCA", detail


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True, help="merchant_embeddings.parquet from fm/embed.py")
    ap.add_argument("--csv", required=True, help="extracted card_transaction.v1.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--copy-out", default="")
    ap.add_argument("--max-points", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    mdf = pl.read_parquet(args.emb)
    emb_cols = [c for c in mdf.columns if c.startswith("emb_")]
    assert len(emb_cols) >= 32, f"unexpectedly few embedding dims: {len(emb_cols)}"
    assert mdf.height > 1000, f"unexpectedly few merchants: {mdf.height}"
    n_merchants = mdf.height
    print(f"[atlas] {n_merchants} merchants x {len(emb_cols)} dims", flush=True)

    mcc_df = modal_mcc_per_merchant(args.csv)
    mdf = mdf.with_columns(
        pl.col("merchant_name").fill_null("NA"), pl.col("merchant_city").fill_null("NA")
    ).join(mcc_df, on=["merchant_name", "merchant_city"], how="left")
    groups = np.array([mcc_group(v) for v in mdf["mcc"].to_list()])
    n_unmatched = int((groups == "Unknown").sum())
    print(f"[atlas] MCC matched for {n_merchants - n_unmatched}/{n_merchants} merchants", flush=True)

    X = mdf.select(emb_cols).to_numpy().astype(np.float32)
    xy, method, detail = project_2d(X, args.seed)
    assert np.all(np.isfinite(xy)), "non-finite projection coordinates"

    # pseudonymous sample: stratified by mcc_group, proportional with >=1 per group
    rng = np.random.default_rng(args.seed)
    n_txns = mdf["n_txns_pre_cut"].to_numpy().astype(np.float64)
    uniq = sorted(set(groups.tolist()))
    take_idx = []
    for g in uniq:
        gi = np.flatnonzero(groups == g)
        k = max(1, int(round(args.max_points * len(gi) / n_merchants)))
        take_idx.append(rng.choice(gi, size=min(k, len(gi)), replace=False))
    take = np.concatenate(take_idx)
    if len(take) > args.max_points:
        take = rng.choice(take, size=args.max_points, replace=False)
    sz = np.log1p(n_txns[take])
    sz = 1.0 + 5.0 * (sz - sz.min()) / max(1e-9, sz.max() - sz.min())

    points = [
        {"x": round(float(xy[i, 0]), 3), "y": round(float(xy[i, 1]), 3),
         "mcc_group": str(groups[i]), "size": round(float(s), 2)}
        for i, s in zip(take, sz)
    ]
    out = {
        "seed": args.seed,
        "versions": {**versions_dict(), **({"umap": detail["umap"]} if "umap" in detail else {})},
        "generated_by": "scripts/scale/atlas_build.py --check-able",
        "data_sources": [
            {"name": "IBM TabFormer card_transaction.v1.csv (synthetic)",
             "url": TABFORMER_URL, "sha256": TABFORMER_SHA256},
            {"name": "merchant_embeddings.parquet (scripts/fm/embed.py, pre-cut pooled, backbone job)",
             "url": "cluster:$HOME/amex-oneloop/merchant_embeddings.parquet", "sha256": "n/a (intermediate)"},
        ],
        "labels": ["synthetic"],
        "method": method,
        "method_detail": detail,
        "points": points,
        "n_merchants": int(n_merchants),
        "n_mcc_unmatched": n_unmatched,
        "colors_by": "mcc_group",
        "pseudonymized": True,
    }
    atomic_write_json(args.out, out)
    if args.copy_out:
        Path(args.copy_out).parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.copy_out, out)
    print(f"[atlas] {len(points)} points ({method}) over {n_merchants} merchants -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
