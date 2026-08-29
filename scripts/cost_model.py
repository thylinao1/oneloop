#!/usr/bin/env python3
"""Cost and return model for One Loop (fix-limits workstream W-COST, limitation L-D).

Every compute quantity is MEASURED: parsed from the committed Slurm accounting dump
at results/cache/sacct_amex_2026-08-24b.txt (jobs named amex-* only), plus row counts
read from the job log amex-merchant-749089.out and two filesystem mtimes that bracket
the embedding phase of that job. Every price is a DECLARED ASSUMPTION with provider,
URL and retrieval date. The two are never mixed: each leaf in the output carries a
basis flag, and the ROI section says in words which side of the ratio is which.

Usage:
  python3 scripts/cost_model.py           # write results/cost_model.json
  python3 scripts/cost_model.py --check   # recompute and compare every value
"""

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "cache" / "sacct_amex_2026-08-24b.txt"
OUT = ROOT / "results" / "cost_model.json"

RETRIEVED = "2026-08-24"

# ---------------------------------------------------------------- prices ----
# DECLARED ASSUMPTIONS. None of these numbers is measured. Each carries its
# provider page and the date it was read. Mappings are conservative: where the
# cluster hardware is a slice, an NVL variant, or an older card with no current
# listing, it is priced at or above the nearest full modern card, which can only
# overstate the cost.

PRICE_BASIS = "declared_assumption"

GPU_PRICES = {
    "a100-40": {
        "usd_per_gpu_hour": 1.99,
        "maps_to": "A100 40GB PCIe/SXM on-demand",
        "provider": "Lambda",
        "url": "https://lambda.ai/pricing",
        "retrieved": RETRIEVED,
        "basis": PRICE_BASIS,
    },
    "a100-80": {
        "usd_per_gpu_hour": 2.79,
        "maps_to": "A100 SXM 80GB on-demand (8x instance rate per GPU, the only 80GB tier Lambda lists)",
        "provider": "Lambda",
        "url": "https://lambda.ai/pricing",
        "retrieved": RETRIEVED,
        "basis": PRICE_BASIS,
    },
    "h100-47": {
        "usd_per_gpu_hour": 4.29,
        "maps_to": ("full H100 SXM 1x on-demand. The cluster resource is a 47GB MIG "
                    "slice, roughly half an H100, priced here as a whole card. "
                    "Conservative: overstates cost."),
        "provider": "Lambda",
        "url": "https://lambda.ai/pricing",
        "retrieved": RETRIEVED,
        "basis": PRICE_BASIS,
    },
    "h100-96": {
        "usd_per_gpu_hour": 4.29,
        "maps_to": ("full H100 SXM 1x on-demand. The cluster card is an H100 NVL 96GB "
                    "(the job log prints 'NVIDIA H100 NVL', 95830 MiB); priced at the "
                    "highest listed H100 tier. Conservative: at or above market."),
        "provider": "Lambda",
        "url": "https://lambda.ai/pricing",
        "retrieved": RETRIEVED,
        "basis": PRICE_BASIS,
    },
    "h200-141": {
        "usd_per_gpu_hour": 50.44 / 8,
        "maps_to": ("HGX H200 141GB, 8-GPU instance at $50.44 per instance-hour, "
                    "divided by 8. Lambda does not list H200 on-demand, so the price "
                    "comes from CoreWeave, which sits above the specialist-cloud "
                    "range. Conservative: overstates cost."),
        "provider": "CoreWeave",
        "url": "https://www.coreweave.com/pricing",
        "retrieved": RETRIEVED,
        "basis": PRICE_BASIS,
    },
    "nv": {
        "usd_per_gpu_hour": 1.99,
        "maps_to": ("A100 40GB price. The gres 'nv' nodes carry far older cards, "
                    "measured via 'scontrol show node' on 2026-08-24: xgpe1/xgpe2 = "
                    "Titan RTX, xgpd0 = Titan V, xgpf4 = T4. None has a current "
                    "listing on a major on-demand page; each rents well below an "
                    "A100 wherever it is still offered. Conservative: overstates."),
        "provider": "Lambda (price), NUS SoC scontrol (hardware identity)",
        "url": "https://lambda.ai/pricing",
        "retrieved": RETRIEVED,
        "basis": PRICE_BASIS,
    },
}

CPU_PRICE = {
    "usd_per_core_hour": 0.714 / 16,
    "maps_to": ("AWS EC2 c7i.4xlarge on-demand, Linux, us-east-1: $0.714 per hour "
                "for 16 vCPU, so $0.0446 per vCPU-hour. Read from the Vantage "
                "mirror of AWS published pricing."),
    "provider": "AWS via instances.vantage.sh",
    "url": "https://instances.vantage.sh/aws/ec2/c7i.4xlarge",
    "retrieved": RETRIEVED,
    "basis": PRICE_BASIS,
}

# ------------------------------------------------- measured serving inputs ----
# MEASURED, with the caveat stated where they are used: the row counts are printed
# in the job log; the wall time is not (the log has no per-phase timestamps), so the
# embedding phase is bracketed by two filesystem mtimes of artifacts that job wrote.

SERVING_MEASURED = {
    "job": "amex-merchant (Slurm 749089), one H100 NVL 96GB (gres h100-96)",
    "log": "~/amex-oneloop/logs/amex-merchant-749089.out on the NUS SoC cluster",
    "n_merchant_embeddings": 238615,     # log line 1589
    "n_window_chunks_pooled": 1415720,   # log lines 1568-1585
    "n_asof_rows_embedded": 1100000,     # log lines 1590-1604
    "phase_start": "2026-08-22T23:47:23.311510+08:00",
    "phase_start_source": ("mtime of ~/amex-oneloop/scale/merchant-axis/ckpt/"
                           "pretrain_summary.json, the last artifact written before "
                           "the embedding phase (ls --full-time, 2026-08-24)"),
    "phase_end": "2026-08-22T23:55:28.065030+08:00",
    "phase_end_source": ("mtime of ~/amex-oneloop/merchant_embeddings_v2.parquet, "
                         "written when the embedding phase persisted its output "
                         "(ls --full-time, 2026-08-24)"),
    "basis": "measured",
}

SERVING_TARGET_MERCHANTS = 1_000_000

# ------------------------------------------------------- value model inputs ----
# DECLARED ASSUMPTIONS, restated from copy/value-model.md (the committed value
# model). The arithmetic below mirrors that file exactly; nothing here is measured.

VALUE_LANES_SOURCE = "copy/value-model.md"

VALUE_INPUTS = {
    # scenario: (markets, lane A [universe, uplift, billed_per_merchant, mdr],
    #            lane B [campaigns_per_partner, billed_per_campaign, mdr],
    #            lane C [budget, share, yield_gain])
    "conservative": {
        "markets": 3,
        "lane_a": {"universe": 30000, "uplift": 0.003, "billed": 50000, "mdr": 0.015},
        "lane_b": {"campaigns": 4, "billed": 1_000_000, "mdr": 0.015},
        "lane_c": {"budget": 3_000_000, "share": 0.15, "gain": 0.10},
    },
    "base": {
        "markets": 8,
        "lane_a": {"universe": 50000, "uplift": 0.008, "billed": 100000, "mdr": 0.02},
        "lane_b": {"campaigns": 6, "billed": 3_000_000, "mdr": 0.02},
        "lane_c": {"budget": 5_000_000, "share": 0.20, "gain": 0.15},
    },
    "stretch": {
        "markets": 15,
        "lane_a": {"universe": 70000, "uplift": 0.015, "billed": 200000, "mdr": 0.025},
        "lane_b": {"campaigns": 10, "billed": 6_000_000, "mdr": 0.025},
        "lane_c": {"budget": 8_000_000, "share": 0.30, "gain": 0.20},
    },
}

HOURS_PER_YEAR = 8760

# ------------------------------------------------------------------ parsing ----

ELAPSED_RE = re.compile(r"^(?:(\d+)-)?(\d+):(\d{2}):(\d{2})$")


def elapsed_seconds(s: str) -> int:
    m = ELAPSED_RE.match(s)
    if not m:
        raise ValueError(f"unparseable Elapsed {s!r}")
    d, h, mi, se = (int(x) if x else 0 for x in m.groups())
    return ((d * 24 + h) * 60 + mi) * 60 + se


def parse_tres(tres: str):
    """AllocTRES -> (gpu_type or None, gpu_count, cpu_count)."""
    gpu_type, gpu_count, cpu = None, 0, 0
    if not tres:
        return gpu_type, gpu_count, cpu
    for tok in tres.split(","):
        if tok.startswith("cpu="):
            cpu = int(tok[4:])
        elif tok.startswith("gres/gpu:"):
            name, _, cnt = tok[len("gres/gpu:"):].partition("=")
            gpu_type, gpu_count = name, int(cnt)
    return gpu_type, gpu_count, cpu


def load_jobs():
    text = RAW.read_text()
    lines = [ln for ln in text.splitlines() if ln.strip()]
    header = lines[0].split("|")
    assert header[:3] == ["JobID", "JobName", "Elapsed"], header
    jobs = []
    for ln in lines[1:]:
        f = ln.split("|")
        name = f[1]
        if not name.startswith("amex-"):
            # the committed dump is already filtered; this guard keeps it that way
            continue
        gpu_type, gpu_count, cpu = parse_tres(f[3])
        state = f[4].split()[0]
        jobs.append({
            "job_id": f[0],
            "name": name,
            "elapsed_seconds": elapsed_seconds(f[2]),
            "gpu_type": gpu_type,
            "gpu_count": gpu_count,
            "cpu_count": cpu,
            "state": state,
            "bucket": "completed" if state == "COMPLETED" else "failed_or_cancelled",
            "node": f[6] if len(f) > 6 else "",
        })
    return jobs


# ------------------------------------------------------------------- model ----

def lane_values(sc: dict):
    a = (sc["markets"] * sc["lane_a"]["universe"] * sc["lane_a"]["uplift"]
         * sc["lane_a"]["billed"] * sc["lane_a"]["mdr"])
    b = (sc["markets"] * sc["lane_b"]["campaigns"] * sc["lane_b"]["billed"]
         * sc["lane_b"]["mdr"])
    c = sc["markets"] * sc["lane_c"]["budget"] * sc["lane_c"]["share"] * sc["lane_c"]["gain"]
    return a, b, c


def build():
    jobs = load_jobs()
    n_completed = sum(1 for j in jobs if j["bucket"] == "completed")
    n_failed = len(jobs) - n_completed

    # -- measured quantities --
    gpu_hours = {}          # type -> {completed, failed_or_cancelled}
    cpu_only_core_hours = {"completed": 0.0, "failed_or_cancelled": 0.0}
    for j in jobs:
        hrs = j["elapsed_seconds"] / 3600.0
        if j["gpu_type"] is not None:
            d = gpu_hours.setdefault(j["gpu_type"],
                                     {"completed": 0.0, "failed_or_cancelled": 0.0})
            d[j["bucket"]] += hrs * j["gpu_count"]
        else:
            cpu_only_core_hours[j["bucket"]] += hrs * j["cpu_count"]

    for d in gpu_hours.values():
        d["total"] = d["completed"] + d["failed_or_cancelled"]
    cpu_only_core_hours["total"] = (cpu_only_core_hours["completed"]
                                    + cpu_only_core_hours["failed_or_cancelled"])
    total_gpu_hours = sum(d["total"] for d in gpu_hours.values())

    # -- priced cost (measured hours x declared-assumption prices) --
    cost_by_type = {}
    for t, d in sorted(gpu_hours.items()):
        p = GPU_PRICES[t]["usd_per_gpu_hour"]
        cost_by_type[t] = {
            "gpu_hours": d,
            "usd_per_gpu_hour": p,
            "usd_completed": d["completed"] * p,
            "usd_failed_or_cancelled": d["failed_or_cancelled"] * p,
            "usd_total": d["total"] * p,
        }
    cpu_cost = {
        "core_hours": cpu_only_core_hours,
        "usd_per_core_hour": CPU_PRICE["usd_per_core_hour"],
        "usd_completed": cpu_only_core_hours["completed"] * CPU_PRICE["usd_per_core_hour"],
        "usd_failed_or_cancelled": (cpu_only_core_hours["failed_or_cancelled"]
                                    * CPU_PRICE["usd_per_core_hour"]),
        "usd_total": cpu_only_core_hours["total"] * CPU_PRICE["usd_per_core_hour"],
    }
    gpu_usd_total = sum(c["usd_total"] for c in cost_by_type.values())
    gpu_usd_failed = sum(c["usd_failed_or_cancelled"] for c in cost_by_type.values())
    usd_total = gpu_usd_total + cpu_cost["usd_total"]
    usd_failed = gpu_usd_failed + cpu_cost["usd_failed_or_cancelled"]

    # -- serving projection --
    t0 = datetime.fromisoformat(SERVING_MEASURED["phase_start"])
    t1 = datetime.fromisoformat(SERVING_MEASURED["phase_end"])
    phase_s = (t1 - t0).total_seconds()
    merchants_per_s = SERVING_MEASURED["n_merchant_embeddings"] / phase_s
    proj_hours = SERVING_TARGET_MERCHANTS / merchants_per_s / 3600.0
    proj_usd = proj_hours * GPU_PRICES["h100-96"]["usd_per_gpu_hour"]

    # -- value lanes (declared assumptions, restated) --
    scenarios = {}
    for name, sc in VALUE_INPUTS.items():
        a, b, c = lane_values(sc)
        scenarios[name] = {
            "lane_a_whitespace_usd_per_year": a,
            "lane_b_offers_usd_per_year": b,
            "lane_c_corridor_usd_per_year": c,
            "total_usd_per_year": a + b + c,
            "basis": PRICE_BASIS,
        }

    # -- payback --
    payback = {}
    for name, sc in scenarios.items():
        annual = sc["total_usd_per_year"]
        frac = usd_total / annual
        payback[name] = {
            "one_off_cost_as_fraction_of_one_year": frac,
            "hours_of_year_at_that_run_rate": frac * HOURS_PER_YEAR,
        }

    out = {
        "generated_by": "scripts/cost_model.py --check-able",
        "what_this_is": ("the cost and return lane the submission did not have: every "
                         "compute quantity measured from Slurm accounting and job "
                         "artifacts, every price a declared assumption with provider, "
                         "URL and retrieval date, and a payback statement against the "
                         "committed declared-assumption value model"),
        "basis_legend": {
            "measured": ("read from Slurm accounting (sacct), a job log, or the "
                         "cluster filesystem; reproducible from the committed dump"),
            "declared_assumption": ("chosen and cited, never measured; prices, "
                                    "mappings and every value-model input"),
        },
        "data_sources": [
            {
                "name": "Slurm accounting dump, amex-* jobs, 2026-08-10 through 2026-08-24",
                "path": "results/cache/sacct_amex_2026-08-24b.txt",
                "sha256": hashlib.sha256(RAW.read_bytes()).hexdigest(),
                "command": ("ssh soc \"sacct -S 2026-08-10 -E now -u $USER "
                            "--format=JobID,JobName%32,Elapsed,AllocTRES%70,State,End,"
                            "NodeList%20 -X -P\" filtered to the header row plus rows "
                            "whose JobName starts with amex-; rows kept verbatim"),
                "retrieved": RETRIEVED,
                "basis": "measured",
            },
            {
                "name": "merchant-axis job log (row counts) and artifact mtimes (phase bracket)",
                "path": SERVING_MEASURED["log"],
                "retrieved": RETRIEVED,
                "basis": "measured",
            },
            {
                "name": "bottom-up value model (three lanes, three scenarios)",
                "path": VALUE_LANES_SOURCE,
                "basis": PRICE_BASIS,
            },
        ],
        "jobs": {
            "n_jobs": len(jobs),
            "n_completed": n_completed,
            "n_failed_or_cancelled": n_failed,
            "included": ("every Slurm job named amex-* on the NUS SoC cluster in the "
                         "window, including failed, cancelled and out-of-memory runs; "
                         "real R&D compute includes the runs that did not work"),
            "excluded": "all jobs of other projects on the same account",
            "basis": "measured",
            "per_job": [
                {k: j[k] for k in ("job_id", "name", "state", "bucket", "gpu_type",
                                   "gpu_count", "cpu_count", "elapsed_seconds", "node")}
                for j in jobs
            ],
        },
        "measured_compute": {
            "gpu_hours_by_type": {t: dict(d, basis="measured", unit="GPU-hours")
                                  for t, d in sorted(gpu_hours.items())},
            "total_gpu_hours": {"value": total_gpu_hours, "basis": "measured",
                                "unit": "GPU-hours"},
            "cpu_only_core_hours": dict(cpu_only_core_hours, basis="measured",
                                        unit="core-hours"),
            "note": ("CPU cores and memory allocated alongside a GPU are not priced "
                     "separately: the cited per-GPU-hour prices are for instances that "
                     "bundle vCPUs and RAM with the card"),
        },
        "price_assumptions": {
            "gpu": GPU_PRICES,
            "cpu_core": CPU_PRICE,
            "all_prices_are": ("declared assumptions: public on-demand rates read on "
                               f"{RETRIEVED}, excluding tax, never presented as "
                               "measured"),
        },
        "one_off_rnd_cost": {
            "by_gpu_type_usd": cost_by_type,
            "cpu_only_usd": cpu_cost,
            "gpu_usd_total": gpu_usd_total,
            "usd_total_market_equivalent": usd_total,
            "usd_failed_or_cancelled_runs": usd_failed,
            "usd_cash_actually_paid": 0.0,
            "which_is_which": ("usd_cash_actually_paid is the real cash outlay: the "
                               "training ran on a university cluster free to the team. "
                               "usd_total_market_equivalent is what the same measured "
                               "hours would have cost at the cited public on-demand "
                               "rates, failed runs included; it is the number to use "
                               "when the cluster subsidy is stripped out."),
            "basis": "measured quantities x declared_assumption prices",
        },
        "serving_projection": {
            "measured_inputs": SERVING_MEASURED,
            "embedding_phase_seconds": phase_s,
            "merchants_per_second_floor": merchants_per_s,
            "target_merchants": {"value": SERVING_TARGET_MERCHANTS,
                                 "basis": "chosen_target",
                                 "note": "illustrative scale, not measured and not a price"},
            "projected_gpu_hours_ceiling": proj_hours,
            "projected_usd_ceiling": proj_usd,
            "price_used": "h100-96 mapping above (declared assumption)",
            "why_floor_and_ceiling": ("the log prints row counts but no per-phase "
                                      "timestamps, so the phase wall time is bracketed "
                                      "by two artifact mtimes. That window also holds "
                                      "checkpoint selection, an as-of embedding pass "
                                      "over 1,100,000 rows and a 454MB parquet copy, "
                                      "all charged here to the 238,615 merchant "
                                      "embeddings alone. The rate is therefore a floor, "
                                      "the projected cost a ceiling."),
            "basis": "measured quantities and timing x declared_assumption price",
        },
        "value_lanes": {
            "source": VALUE_LANES_SOURCE,
            "inputs_restated": VALUE_INPUTS,
            "scenarios": scenarios,
            "what_these_are": ("the committed declared-assumption planning model, "
                               "restated with its own arithmetic: every input is an "
                               "assumption or a cited public number, none is a "
                               "measurement of One Loop in operation. The offers lane "
                               "is holdout-certifiable once campaigns run; the "
                               "corridor lane's yield-gain input is flagged in the "
                               "value model as its least anchored number."),
            "basis": PRICE_BASIS,
        },
        "roi": {
            "shape": ("annual declared-assumption value vs a one-off measured-quantity "
                      "cost, so this is a payback statement and not a return "
                      "percentage: the numerator is a planning model, the denominator "
                      "is measured hours at cited prices"),
            "payback": payback,
            "statement": (f"At the conservative scenario's declared-assumption "
                          f"${scenarios['conservative']['total_usd_per_year']:,.0f} per year, "
                          f"the one-off market-price equivalent of all R&D compute "
                          f"(failed runs included) equals what that scenario models in "
                          f"about {payback['conservative']['hours_of_year_at_that_run_rate'] * 60:.0f} "
                          f"minutes of a year; at the base scenario, in "
                          f"about {payback['base']['hours_of_year_at_that_run_rate'] * 60:.1f} "
                          f"minutes. The ratio is this "
                          "lopsided because the denominator is small, not because the "
                          "numerator is proven: the value side remains a "
                          "declared-assumption model, and no return here is measured."),
        },
        "caveats": [
            "Every price is a declared assumption read on one day from one provider "
            "page; on-demand prices move and exclude tax.",
            "h100-47 is a 47GB MIG slice priced as a full H100; h100-96 is an H100 NVL "
            "priced at the top listed H100 SXM tier; the 'nv' jobs ran on Titan RTX, "
            "Titan V and T4 cards priced at the A100 40GB rate. Each mapping "
            "overstates cost.",
            "Elapsed hours come from Slurm job elapsed time, which includes any "
            "in-job idle time; nothing is subtracted.",
            "The window starts 2026-08-10; amex-* jobs first appear 2026-08-22, and "
            "no amex-* job exists before the window (job ids begin at 748860).",
            "The serving projection's wall time is bracketed by artifact mtimes, not "
            "in-log timestamps; the caveat inside serving_projection states what the "
            "bracket includes and why the cost is a ceiling.",
            "Laptop-side development, staff time and data egress are not costed; "
            "this model prices cluster compute only.",
            "The value lanes are not validated outcomes. Payback against them "
            "inherits every assumption they carry, including the corridor lane's "
            "least-anchored yield-gain input.",
        ],
        "required_sentence": (
            f"Building One Loop took a measured {total_gpu_hours:.1f} GPU-hours and "
            f"{cpu_only_core_hours['total']:.0f} CPU-core-hours "
            f"on a university cluster: cash cost zero, about "
            f"US${usd_total:.0f} at cited public "
            "on-demand rates with every failed run included, and refreshing "
            f"embeddings for one million merchants projects to about "
            f"US${proj_usd:.2f} per run "
            "at the same rates; the annual value it stands against is a "
            "declared-assumption planning model, so the payback ratio is an "
            "assumption, not a measurement."
        ),
        "check": {
            "command": "python3 scripts/cost_model.py --check",
            "tolerance": 1e-6,
        },
    }
    return out


# ------------------------------------------------------------------- check ----

def _compare(a, b, path, tol=1e-6, bad=None):
    if bad is None:
        bad = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            if k == "generated_by":
                continue
            if k not in a or k not in b:
                bad.append(f"{path}/{k}: present in only one file")
            else:
                _compare(a[k], b[k], f"{path}/{k}", tol, bad)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            bad.append(f"{path}: length {len(a)} vs {len(b)}")
        else:
            for i, (x, z) in enumerate(zip(a, b)):
                _compare(x, z, f"{path}/{i}", tol, bad)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if abs(float(a) - float(b)) > tol:
            bad.append(f"{path}: {a} vs {b}")
    elif a != b:
        bad.append(f"{path}: {a!r} vs {b!r}")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="recompute from the committed dump and compare every value "
                         "against results/cost_model.json")
    args = ap.parse_args()
    out = build()
    if args.check:
        prev = json.loads(OUT.read_text())
        bad = _compare(prev, out, "")
        if bad:
            print("CHECK FAILED", flush=True)
            for b in bad[:40]:
                print("  ", b, flush=True)
            sys.exit(1)
        print(f"CHECK OK ({OUT.relative_to(ROOT)} reproduces within 1e-6)", flush=True)
        return
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"[run] wrote {OUT.relative_to(ROOT)}", flush=True)
    print(f"[run] total GPU-hours {out['measured_compute']['total_gpu_hours']['value']:.3f}, "
          f"market equivalent ${out['one_off_rnd_cost']['usd_total_market_equivalent']:.2f} "
          f"(failed runs ${out['one_off_rnd_cost']['usd_failed_or_cancelled_runs']:.2f}), "
          f"cash $0", flush=True)


if __name__ == "__main__":
    main()
