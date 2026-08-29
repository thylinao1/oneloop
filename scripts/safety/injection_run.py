"""SAFE-C, stage 2: run the FROZEN attack corpus through the REAL narrative pipeline.

Not a mock. This imports the shipped prompts, the shipped sampling parameters, the
shipped checker parser and the shipped deterministic matcher straight out of
scripts/narratives/, so what is measured is the gate as built:

  writer   GEN_SYSTEM   + json.dumps(facts)            -> narrative
  layer 1  faithcheck.unsupported_numerals(narrative, facts)   deterministic, no model
  layer 2  CHECK_SYSTEM + json.dumps(facts) + narrative -> parse_checker -> flags
  strict pass = no unsupported numeral AND no cross-exam flag

Two arms in one job, one model load, identical sampling:
  clean     the 30 committed bundles, no injection
  attacked  the 210 frozen probes

Writes only raw model output plus the two layers' raw verdicts. All scoring,
every rate and every interval happens later on CPU in injection_redteam.py, so
the model never sees a metric and the metric code never sees a GPU.

Usage (inside the sbatch job):
  python -u injection_run.py --corpus injection_corpus.json \
      --out safety_injection_raw.json --model Qwen/Qwen3-32B --tp 2
"""

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "narratives"))
from faithcheck import collect_fact_numbers_string_scan  # noqa: E402
from faithcheck import unsupported_numerals as _gate_unsupported_numerals  # noqa: E402

from generate import (  # noqa: E402
    CHECK_SYSTEM,
    GEN_SYSTEM,
    SEED,
    build_check_prompts,
    build_prompts,
    parse_checker,
)
# The red-team run scores layer one as it stood when the published measurement
# was taken, with the allowed pool built by scanning every fact string. The gate
# no longer does that (scripts/narratives/faithcheck.py).
def unsupported_numerals(narrative, facts):
    return _gate_unsupported_numerals(
        narrative, facts, pool_fn=collect_fact_numbers_string_scan)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    args = ap.parse_args()

    import torch
    # NUS-COMPUTE.md Hopper cuDNN guard, same line generate.py runs.
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)

    import transformers
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    corpus = json.loads(Path(args.corpus).read_text())
    clean = corpus["clean_bundles"]
    probes = corpus["probes"]

    # One flat list: the clean arm first, then every attacked probe, in corpus order.
    items = []
    for b in clean:
        items.append({"arm": "clean", "key": b["id"], "bundle_id": b["id"],
                      "head": b["head"], "attack_class": None, "facts": b["facts"]})
    for p in probes:
        items.append({"arm": "attacked", "key": p["probe_id"], "bundle_id": p["bundle_id"],
                      "head": p["head"], "attack_class": p["attack_class"],
                      "facts": p["facts"]})
    print(f"[inj] {len(clean)} clean + {len(probes)} attacked = {len(items)} runs",
          flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, dtype="bfloat16", seed=SEED,
              max_model_len=args.max_model_len,
              tensor_parallel_size=args.tp,
              gpu_memory_utilization=args.gpu_mem_util)
    # Identical to the shipped run (generate.py:113-114).
    gen_sp = SamplingParams(temperature=0.0, seed=SEED, max_tokens=350)
    chk_sp = SamplingParams(temperature=0.0, seed=SEED, max_tokens=600)

    bundles = [{"facts": it["facts"]} for it in items]

    t0 = time.time()
    gen_out = llm.generate(build_prompts(tok, bundles), gen_sp)
    narratives = [o.outputs[0].text.strip() for o in gen_out]
    t_gen = time.time() - t0
    print(f"[inj] generated {len(narratives)} in {t_gen:.1f}s", flush=True)

    # layer one, deterministic, no model in the loop. Run against the facts the
    # gate was actually handed (injected where injected), and separately against
    # the pre-injection facts so the pool-pollution effect is visible in the raw.
    layer1 = [unsupported_numerals(n, it["facts"]) for n, it in zip(narratives, items)]
    clean_facts_by_bundle = {b["id"]: b["facts"] for b in clean}
    layer1_vs_original = [unsupported_numerals(n, clean_facts_by_bundle[it["bundle_id"]])
                          for n, it in zip(narratives, items)]

    t1 = time.time()
    chk_out = llm.generate(build_check_prompts(tok, bundles, narratives), chk_sp)
    chk_raw = [o.outputs[0].text.strip() for o in chk_out]
    layer2 = [parse_checker(t) for t in chk_raw]
    t_chk = time.time() - t1
    print(f"[inj] cross-examined {len(chk_raw)} in {t_chk:.1f}s", flush=True)

    runs = []
    for it, narr, l1, l1o, raw, l2 in zip(
            items, narratives, layer1, layer1_vs_original, chk_raw, layer2):
        runs.append({
            "key": it["key"], "arm": it["arm"], "bundle_id": it["bundle_id"],
            "head": it["head"], "attack_class": it["attack_class"],
            "narrative": narr,
            "layer1_unsupported_numerals": l1,
            "layer1_unsupported_numerals_vs_original_facts": l1o,
            "layer2_raw": raw,
            "layer2_flags": l2,
            "strict_pass": (not l1) and (not l2),
        })

    out = {
        "seed": SEED,
        "versions": {"python": sys.version.split()[0], "vllm": vllm.__version__,
                     "torch": torch.__version__,
                     "transformers": transformers.__version__},
        "generated_by": "scripts/safety/injection_run.py",
        "model": args.model,
        "served": "self-hosted vLLM on NUS GPU",
        "tensor_parallel_size": args.tp,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_mem_util,
        "sampling": {"temperature": 0.0, "seed": SEED,
                     "gen_max_tokens": 350, "check_max_tokens": 600},
        "prompts_are_the_shipped_ones": {
            "gen_system": GEN_SYSTEM, "check_system": CHECK_SYSTEM,
            "note": ("imported from scripts/narratives/generate.py, not restated here, "
                     "so a change to the shipped gate changes this run too")},
        "corpus_file": Path(args.corpus).name,
        "n_runs": len(runs),
        "timing_seconds": {"generation": t_gen, "cross_examination": t_chk},
        "runs": runs,
    }
    Path(args.out).write_text(json.dumps(out, indent=1))
    n_clean_pass = sum(r["strict_pass"] for r in runs if r["arm"] == "clean")
    n_att_pass = sum(r["strict_pass"] for r in runs if r["arm"] == "attacked")
    print(f"[inj] clean strict pass {n_clean_pass}/{len(clean)}; "
          f"attacked strict pass {n_att_pass}/{len(probes)} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
