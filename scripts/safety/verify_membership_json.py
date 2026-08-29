"""verify_membership_json.py, read the SAFE-A result back and re-state it.

Checker separate from maker. This opens results/safety_membership.json, prints the
numbers exactly as stored, and hard-fails on the structural rules committed to for this file:
the envelope, the pointer law, the control law, the FPR floor, the calibration arms, and
the forbidden phrasings. It computes nothing new, so it cannot launder a number.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

FORBIDDEN = ["the model does not leak", "membership inference fails", "no leakage detected"]


def main(path: str) -> int:
    p = Path(path)
    raw = p.read_bytes()
    d = json.loads(raw)
    fails = []

    print(f"file      {p}")
    print(f"bytes     {len(raw)}")
    print(f"sha256    {hashlib.sha256(raw).hexdigest()}")
    print(f"generated {d.get('generated_at')}  by {d.get('generated_by')}")
    print(f"labels    {d.get('labels')}")
    print()

    for k in ("seed", "versions", "generated_by", "data_sources", "labels"):
        if k not in d:
            fails.append(f"envelope missing {k}")
    if "synthetic" not in d.get("labels", []):
        fails.append("labels must carry 'synthetic'")
    for s in d.get("data_sources", []):
        if not s.get("sha256"):
            fails.append(f"data source without sha256: {s.get('name')}")
    if not d.get("mechanism_not_exposure"):
        fails.append("mechanism_not_exposure sentence missing")
    if d.get("zero_labels", {}).get("fraud_labels_opened") is not False:
        fails.append("label-free claim not recorded as False")

    txt = json.dumps(d).lower()
    for f in FORBIDDEN:
        # the guard block is allowed to NAME the forbidden phrasings; nothing else may use them
        allowed = json.dumps(d.get("interpretation_guard", {})).lower()
        if txt.count(f) > allowed.count(f):
            fails.append(f"forbidden phrasing used outside the guard block: {f!r}")

    pop = d["population"]
    print("POPULATION")
    print(f"  primary stratum            {pop['stratum_primary']}")
    print(f"  substitution               {pop['primary_substitution']}")
    print(f"  m chosen                   {pop['m_chosen']}  from {pop['m_candidates_non_member_accounts_with_at_least_m_rows']}")
    print(f"  scored rows                {pop['n_scored_rows']}")
    print(f"  member rows sampled        {pop['n_member_rows_sampled']}")
    print(f"  contamination-gradient     {pop['n_member_rows_contamination_gradient_only']}")
    print(f"  non-member rows            {pop['n_non_member_rows']}")
    print(f"  all scored rows post-cut   {pop['all_scored_rows_post_cut']}")
    print(f"  asserted counts            {d['membership_rule']['asserted_counts']}")
    if pop["all_scored_rows_post_cut"] is not True:
        fails.append("a scored row is not post-cut")
    print()

    prim = pop["stratum_primary"]
    for name, st in d["strata"].items():
        tag = "PRIMARY" if st["is_primary"] else "secondary"
        print(f"STRATUM {name}  [{tag}]")
        print(f"  member {st['n_member_rows']} rows / {st['n_member_accounts']} accounts   "
              f"non-member {st['n_non_member_rows']} rows / {st['n_non_member_accounts']} accounts")
        rf = st["fpr_floor"]["row_level"]["smallest_measurable_fpr"]
        af = st["fpr_floor"]["account_level"]["smallest_measurable_fpr"]
        print(f"  FPR floor: row {rf:.3e}   account {af:.6f}")
        for arm, v in st["arms"].items():
            def ci(k):
                c = v.get(k)
                return "no interval" if c is None else f"[{c[0]:.4f}, {c[1]:.4f}]"
            print(f"    {arm:36s} row_auc {v['row_auc']:.4f} {ci('row_auc_ci')}"
                  f"  acct_auc {v['account_auc']:.4f} {ci('account_auc_ci')}")
            print(f"    {'':36s} TPR@FPR<=1e-4 {v['row_tpr_at_fpr_0.0001']:.5f} "
                  f"(achieved {v['row_achieved_fpr_0.0001']:.3e})"
                  f"   TPR@FPR<=1e-3 {v['row_tpr_at_fpr_0.001']:.5f} "
                  f"(achieved {v['row_achieved_fpr_0.001']:.3e})"
                  f"   acct TPR@floor {v['account_tpr_at_fpr_floor']:.4f} "
                  f"(achieved {v['account_achieved_fpr']:.6f})")
        mde = st["minimum_detectable_effect"]
        print(f"  minimum detectable effect (pooled sd {mde['pooled_sd']:.4f}):")
        for dl, e in mde["by_delta"].items():
            c = e["account_auc_ci"]
            print(f"    planted shift {dl:>5s} sd -> account AUC {e['account_auc']:.4f} "
                  f"{'no interval' if c is None else f'[{c[0]:.4f}, {c[1]:.4f}]'}")
        print()

    print("COMPARISONS (primary stratum only)")
    for k, e in d["comparisons_by_key"].items():
        for met in ("row_auc", "account_auc"):
            c = e[met]["ci"]
            print(f"  {k}  {met}: {e[met]['a_value']:.4f} vs {e[met]['b_value']:.4f}  "
                  f"diff {e[met]['difference']:+.4f}  "
                  f"{'no interval' if c is None else f'[{c[0]:+.4f}, {c[1]:+.4f}]'}  "
                  f"-> {e[met]['direction']}")
        if e not in d["comparisons"]:
            fails.append(f"comparisons_by_key[{k}] is not the same object as its list entry")
    print()

    print("DIAGNOSTICS (not attacks)")
    t = d["diagnostics_not_attacks"]["account_tenure"]
    print(f"  median total rows: member {t['median_total_rows_member_accounts']}  "
          f"non-member {t['median_total_rows_non_member_accounts']}")
    if "overlap_curve_pairs_by_caliper_log10" in t:
        print(f"  tenure overlap, matched pairs by caliper (log10 of account size): "
              f"{t['overlap_curve_pairs_by_caliper_log10']}")
    print(f"  tenure-matched stratum run: {t['matching']['run']}")
    if not t["matching"]["run"]:
        print(f"    reason: {t['matching']['reason']}")
    print("  contamination gradient (pretrained rows inside the scored window):")
    for k, v in d["diagnostics_not_attacks"]["contamination_gradient_by_window"].items():
        if v["row_auc"] is None:
            print(f"    level {k:>2s}: n={v['n_member_rows']:>6d}  {v.get('note','')}")
        else:
            print(f"    level {k:>2s}: n={v['n_member_rows']:>6d}  "
                  f"model row_auc {v['row_auc']:.4f}  control row_auc {v['control_row_auc']:.4f}")
    print()

    # CONTROL LAW: the model arm may never be reported without the control beside it
    st = d["strata"][prim]["arms"]
    for need in ("model_pll_behavioural", "control_unigram_rarity",
                 "calibration_positive_tenure", "calibration_negative_random"):
        if need not in st:
            fails.append(f"arm missing from the primary stratum: {need}")
    pos = st.get("calibration_positive_tenure", {})
    neg = st.get("calibration_negative_random", {})
    if pos and pos["account_auc"] <= 0.5:
        fails.append("positive calibration control did not separate: the pipeline is not "
                     "demonstrated to be able to detect an effect")
    if neg and neg.get("account_auc_ci") and not (neg["account_auc_ci"][0] <= 0.5 <= neg["account_auc_ci"][1]):
        fails.append("negative calibration control excludes 0.5: the evaluation is leaking")

    print("CALIBRATION VERDICT")
    print(f"  positive control account AUC {pos.get('account_auc')}  "
          f"CI {pos.get('account_auc_ci')}   (must be well above 0.5)")
    print(f"  negative control account AUC {neg.get('account_auc')}  "
          f"CI {neg.get('account_auc_ci')}   (CI must cover 0.5)")
    print()

    if fails:
        for f in fails:
            print(f"VERIFY FAILED: {f}")
        return 1
    print(f"VERIFY OK: {len(json.dumps(d))} bytes of JSON, every structural rule held")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "results/safety_membership.json"))
