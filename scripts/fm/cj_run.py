"""cj_run.py - run an UNMODIFIED fm/ stage (pretrain or embed) on a Complete Journey prep.

The strongest form of the replication claim is that the SAME training and embedding code
executed on the real corpus. pretrain.py and embed.py touch field names only through
common.load_prep's equality assert against common.FIELDS, so this launcher installs
CJ_FIELDS (and the CJ forbidden-column set) into the already-imported common module and
then executes the target script byte-for-byte via runpy. No fm/ file is modified; the
patch is process-local and applied before any pipeline code runs.

Leakage: unchanged from the target scripts. pretrain.py still hard-truncates at cut_ts
(prep_cj.py wrote it); embed.py still pools merchants (here: stores) from pre-cut rows
only and builds as-of embeddings strictly before the scored row. The patched
FORBIDDEN_COLUMNS additionally hard-fail if any CJ identifier or coupon column ever
appears in meta['fields'].

Usage: python cj_run.py pretrain --prep ... --ckpt ... [pretrain.py args]
       python cj_run.py embed    --prep ... --ckpt ... [embed.py args]
"""
from __future__ import annotations

import runpy
import sys

from cj_common import patch_common

STAGES = ("pretrain", "embed")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in STAGES:
        sys.exit(f"usage: cj_run.py {{{'|'.join(STAGES)}}} [stage args...]")
    stage = sys.argv[1]
    patch_common()
    sys.argv = [f"{stage}.py"] + sys.argv[2:]
    runpy.run_module(stage, run_name="__main__")


if __name__ == "__main__":
    main()
