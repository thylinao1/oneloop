"""GenAI explanation-layer exhibit: generate partner-facing narratives over real
exhibit facts with a self-hosted vLLM model, then run the faithfulness harness.

Runs on ONE NUS cluster GPU (h200-141 preferred). GenAI explains, never decides
(red line K5). Honesty rider: failures ship with fail_reason; the strict pass
RATE is the exhibit.

Harness layers:
  (a) deterministic numeric check (faithcheck.py) -- every numeral in a
      narrative must match a fact value (1e-6, rounding-tolerant, percent-aware)
  (b) LLM cross-examination -- the same served model is asked, at temperature 0,
      to list every claim not supported by the facts.
Strict pass = zero unsupported numerals AND zero cross-exam flags.

Usage (inside the sbatch job):
  python -u generate.py --inputs narratives_inputs.json --out narratives.json \
      --model Qwen/Qwen3-32B
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from faithcheck import unsupported_numerals  # noqa: E402

SEED = 42

GEN_SYSTEM = (
    "You are the explanation layer of a decision-science system. "
    "Write a 3-4 sentence partner-facing narrative using ONLY the facts below. "
    "Every number you mention must appear in the facts. Do not add causal claims."
)

CHECK_SYSTEM = (
    "You are a strict fact-checking auditor. You will be given FACTS (JSON) and a "
    "NARRATIVE written from them. List every claim in the narrative that is not "
    "supported by the facts (numbers not present in the facts, causal claims, or "
    "statements the facts do not contain). Respond with JSON only, in the form "
    '{"unsupported_claims": ["..."]}. If every claim is supported, respond '
    '{"unsupported_claims": []}.'
)


def build_prompts(tok, bundles):
    gen_prompts = []
    for b in bundles:
        messages = [
            {"role": "system", "content": GEN_SYSTEM},
            {"role": "user", "content": "FACTS:\n" + json.dumps(b["facts"], indent=1)},
        ]
        gen_prompts.append(tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False))
    return gen_prompts


def build_check_prompts(tok, bundles, narratives):
    prompts = []
    for b, narr in zip(bundles, narratives):
        messages = [
            {"role": "system", "content": CHECK_SYSTEM},
            {"role": "user", "content": (
                "FACTS:\n" + json.dumps(b["facts"], indent=1) +
                "\n\nNARRATIVE:\n" + narr)},
        ]
        prompts.append(tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False))
    return prompts


def parse_checker(text):
    """Parse the cross-exam JSON. Unparseable output counts as a flag (honest)."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return ["CHECKER-UNPARSEABLE: " + text[:200].strip()]
    try:
        obj = json.loads(m.group(0))
        claims = obj.get("unsupported_claims", [])
        if not isinstance(claims, list):
            return ["CHECKER-UNPARSEABLE: unsupported_claims not a list"]
        return [str(c) for c in claims]
    except (json.JSONDecodeError, AttributeError):
        return ["CHECKER-UNPARSEABLE: " + text[:200].strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--max-model-len", type=int, default=8192)
    args = ap.parse_args()

    import torch
    # NUS-COMPUTE.md Hopper cuDNN guard (h100/h200 nodes)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)

    import transformers
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    inputs = json.loads(Path(args.inputs).read_text())
    bundles = inputs["bundles"]
    print(f"[narratives] {len(bundles)} fact bundles loaded", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, dtype="bfloat16", seed=SEED,
              max_model_len=args.max_model_len)
    gen_sp = SamplingParams(temperature=0.0, seed=SEED, max_tokens=350)
    chk_sp = SamplingParams(temperature=0.0, seed=SEED, max_tokens=600)

    # --- generation ---
    gen_out = llm.generate(build_prompts(tok, bundles), gen_sp)
    narratives = [o.outputs[0].text.strip() for o in gen_out]
    print(f"[narratives] generated {len(narratives)}", flush=True)

    # --- harness (a): deterministic numeric check ---
    numeric_bad = [unsupported_numerals(n, b["facts"])
                   for n, b in zip(narratives, bundles)]

    # --- harness (b): LLM cross-examination ---
    chk_out = llm.generate(build_check_prompts(tok, bundles, narratives), chk_sp)
    exam_flags = [parse_checker(o.outputs[0].text.strip()) for o in chk_out]

    examples = []
    n_pass = 0
    for b, narr, bad, flags in zip(bundles, narratives, numeric_bad, exam_flags):
        ok = not bad and not flags
        n_pass += ok
        reasons = []
        if bad:
            reasons.append("unsupported numerals: " + ", ".join(bad))
        if flags:
            reasons.append("cross-exam flags: " + " | ".join(flags))
        examples.append({
            "id": b["id"],
            "head": b["head"],
            "input_facts": b["facts"],
            "narrative": narr,
            "numeric_unsupported": bad,
            "cross_exam_flags": flags,
            "verdict": "pass" if ok else "fail",
            "fail_reason": None if ok else "; ".join(reasons),
        })

    out = {
        "seed": SEED,
        "versions": {
            "python": sys.version.split()[0],
            "vllm": vllm.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "generated_by": "scripts/narratives/generate.py (check: scripts/narratives/check.py --check)",
        "data_sources": inputs["data_sources"],
        "labels": ["genai-explanation-layer", "facts-from-committed-results",
                   "explains-never-decides"],
        "model": args.model,
        "served": "self-hosted vLLM on NUS GPU",
        "sampling": {"temperature": 0.0, "seed": SEED,
                     "inputs_generated_by": inputs["generated_by"],
                     "inputs_seed": inputs["seed"]},
        "n_generated": len(narratives),
        "faithfulness": {
            "n_checked": len(examples),
            "n_pass_strict": n_pass,
            "checker": "numeric-claim extraction vs source JSON + LLM cross-check",
            "failures_shown": True,
        },
        "examples": examples,
    }
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"[narratives] strict pass {n_pass}/{len(examples)} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
