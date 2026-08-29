"""Build narratives_inputs.json: ~30 fact bundles sampled from committed results/.

Runs on the Mac (light, stdlib only). Facts ONLY -- no free text beyond
labels/descriptions of what each number is. Deterministic (seed 42).

Usage: python3 scripts/narratives/make_inputs.py
Output: results/cache/narratives_inputs.json
"""

import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEED = 42
N_PER_HEAD = 10


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def corridor_bundles(rng):
    src = ROOT / "results" / "corridor.json"
    d = json.loads(src.read_text())
    rows = rng.sample(d["corridors"], min(N_PER_HEAD, len(d["corridors"])))
    bundles = []
    for r in rows:
        bundles.append({
            "head": "corridor",
            "facts": {
                # The rule below used to read "MASE below 1 beats the seasonal-naive baseline".
                # That is false for this comparison and it produced five narratives asserting the
                # reverse of their own numbers: mase_seasonal_naive is a separately computed
                # figure, so the comparison is model against THAT number, not against 1. Every
                # reversed narrative had a model MASE below 1, so the generator applied the rule
                # it was handed. State the actual rule.
                "what": ("held-out monthly backtest of Singapore inbound visitor arrivals by "
                         "origin market (public proxy for cross-border spend corridors); MASE is "
                         "an error measure and LOWER IS BETTER, so the model beats the seasonal "
                         "naive on a corridor only when mase_model is less than "
                         "mase_seasonal_naive, and comparing either number to 1 says nothing "
                         "about which of the two won"),
                "corridor": r["origin"],
                "region": r["region"],
                "holdout_months": d["holdout_months"],
                "mase_model": r["mase_model"],
                "mase_seasonal_naive": r["mase_seasonal_naive"],
                "mase_model_reconciled": r["mase_model_reconciled"],
                "top3_model_attributions": r["attribution_top3"],
                "attribution_label": d["attribution_label"],
            },
        })
    return bundles, src


def whitespace_bundles(rng):
    src = ROOT / "results" / "whitespace.json"
    d = json.loads(src.read_text())
    pool = d["ranking"][:50]
    rows = rng.sample(pool, N_PER_HEAD)
    bundles = []
    for r in rows:
        bundles.append({
            "head": "whitespace",
            "facts": {
                "what": ("pseudonymized Singapore merchant-signing whitespace bucket, ranked by "
                         "real public signals only (scores in 0-1); sensitivity = this bucket's "
                         "rank if a simulated demand-weighted acceptance-gap signal were added "
                         "at the stated strength"),
                "bucket_label": r["bucket_label"],
                "area": r["area"],
                "category": r["category"],
                "rank_real_signals": r["rank"],
                "n_pois": r["n_pois"],
                "score_real_signals": r["score_real_signals"],
                "reasons": r["reasons"],
                "sensitivity": r["sensitivity"],
            },
        })
    return bundles, src


def hillstrom_bundles(rng):
    src = ROOT / "results" / "uplift.json"
    d = json.loads(src.read_text())
    h = d["hillstrom"]
    segs = h["segments"]
    rows = rng.sample(segs, N_PER_HEAD)
    # deterministic coverage repair: ensure every verdict class present
    for verdict in ("wasted-budget", "hidden-gem", "aligned"):
        if not any(s["verdict"] == verdict for s in rows):
            cand = next(s for s in segs if s["verdict"] == verdict)
            repl = max((s for s in rows if sum(x["verdict"] == s["verdict"] for x in rows) > 1),
                       key=lambda s: s["name"])
            rows[rows.index(repl)] = cand
    bundles = []
    for s in rows:
        bundles.append({
            "head": "uplift",
            "facts": {
                "what": ("customer segment from the randomized Hillstrom e-mail experiment "
                         "(Womens E-Mail vs No E-Mail); ranks are 1=highest among 28 segments; "
                         "response_rank orders segments by a response model, uplift_rank by an "
                         "uplift model, measured_uplift_rank by uplift measured directly from "
                         "the randomized arms; the verdict compares response-model targeting "
                         "against measured uplift"),
                "segment": s["name"],
                "n_treated": s["n_treated"],
                "n_control": s["n_control"],
                "response_rank": s["response_rank"],
                "uplift_rank": s["uplift_rank"],
                "measured_uplift_rank": s["measured_uplift_rank"],
                "measured_visit_uplift_pp": s["measured_visit_uplift_pp"],
                "measured_visit_uplift_se_pp": s["measured_visit_uplift_se_pp"],
                "verdict": s["verdict"],
                "verdict_rule": h["verdict_rule"],
            },
        })
    return bundles, src


def main():
    rng = random.Random(SEED)
    all_bundles = []
    sources = []
    for fn in (corridor_bundles, whitespace_bundles, hillstrom_bundles):
        bundles, src = fn(rng)
        all_bundles.extend(bundles)
        sources.append({
            "name": src.name,
            "url": str(src.relative_to(ROOT)),
            "sha256": sha256(src),
        })
    for i, b in enumerate(all_bundles):
        b["id"] = f"{b['head']}-{i:02d}"
    out = {
        "seed": SEED,
        "versions": {"python": sys.version.split()[0]},
        "generated_by": "scripts/narratives/make_inputs.py",
        "data_sources": sources,
        "labels": ["facts-only", "sampled-from-committed-results"],
        "n_bundles": len(all_bundles),
        "bundles": all_bundles,
    }
    dest = ROOT / "results" / "cache" / "narratives_inputs.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    print(f"wrote {dest} ({len(all_bundles)} bundles)")


if __name__ == "__main__":
    main()
