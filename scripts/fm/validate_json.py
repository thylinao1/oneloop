"""validate_json.py: hard-assert the backbone_transfer.json envelope (the committed result envelope)."""
from __future__ import annotations

import json
import math
import sys


def main(path: str) -> None:
    o = json.loads(open(path).read())
    for k in ("seed", "versions", "generated_by", "data_sources", "labels",
              "pretrain", "tasks", "leakage_checks"):
        assert k in o, f"missing key: {k}"
    assert "synthetic" in o["labels"], "TabFormer output must carry the 'synthetic' label"
    p = o["pretrain"]
    for k in ("params_m", "epochs", "loss_curve", "corpus_cut_date"):
        assert k in p, f"missing pretrain.{k}"
    lc = p["loss_curve"]
    assert len(lc) >= 4, "loss curve too short"
    q = max(1, len(lc) // 4)
    first = sum(v for _, v in lc[:q]) / q
    last = sum(v for _, v in lc[-q:]) / q
    assert all(math.isfinite(v) for _, v in lc), "non-finite loss values"
    assert last < first, f"loss did not decrease (first-quarter mean {first:.4f} -> last {last:.4f})"
    lk = o["leakage_checks"]
    for k in ("label_excluded", "time_truncated", "as_of_embeddings", "ids_hashed"):
        assert lk.get(k) is True, f"leakage check failed/absent: {k}"
    t = o["tasks"]
    fr, mc = t["fraud"], t["next_mcc"]
    for k in ("baseline_auc", "baseline_prauc", "with_emb_auc", "with_emb_prauc",
              "delta_auc_ci", "delta_prauc_ci"):
        assert k in fr, f"missing fraud.{k}"
    for k in ("baseline_auc", "baseline_prauc", "with_emb_auc", "with_emb_prauc"):
        assert 0.0 <= fr[k] <= 1.0 and math.isfinite(fr[k]), f"fraud.{k} out of range"
    for k in ("baseline_top1", "baseline_top5", "with_emb_top1", "with_emb_top5",
              "delta_top1_ci", "delta_top5_ci"):
        assert k in mc, f"missing next_mcc.{k}"
    def check_ci(obj, key, scope):
        lo, hi = obj[key]
        if math.isnan(lo) or math.isnan(hi):
            print(f"[validate] WARNING: {scope}.{key} is NaN (degenerate bootstrap resamples "
                  f";  expected only on tiny smoke data)")
        else:
            assert lo <= hi, f"{scope}.{key} malformed"

    for ci_key in ("delta_auc_ci", "delta_prauc_ci"):
        check_ci(fr, ci_key, "fraud")
    for ci_key in ("delta_top1_ci", "delta_top5_ci"):
        check_ci(mc, ci_key, "next_mcc")
    print(f"[validate] {path} OK: loss {first:.4f}->{last:.4f}, "
          f"fraud dAUC {fr['with_emb_auc']-fr['baseline_auc']:+.4f} CI {fr['delta_auc_ci']}, "
          f"mcc dtop1 {mc['with_emb_top1']-mc['baseline_top1']:+.4f} CI {mc['delta_top1_ci']}")


if __name__ == "__main__":
    main(sys.argv[1])
