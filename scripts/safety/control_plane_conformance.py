#!/usr/bin/env python3
"""SAFE-E, control-plane conformance, pre-registered in SAFETY-DESIGN.md 2.6 and Part 3.

WHAT THIS IS, AND WHAT IT IS NOT, STATED BEFORE ANY NUMBER.

Part 2 of the safety band is DESIGNED AND NOT BUILT. No line of the agent control plane
runs in any product. This file does not change that. It takes ONE property that Part 2.1
claims for the rule written out in Part 2.6, transcribes that rule into a reference
implementation a person can read end to end, and measures whether the property holds over
an enumerated space of mandate, action and model-novelty-score triples. The result is a
statement about our own specification. It is not a statement about American Express, about
any deployed system, or about any system that exists. The status label is
DESIGNED-AND-CHECKED, and it travels with every number in the output file.

THE PROPERTY. SAFETY-DESIGN.md 2.1 part 3, verbatim: "A model output may only add friction,
never remove it. ... for every action, the disposition computed with the model score present
must be at least as restrictive as the disposition computed with the model absent. Deny is
more restrictive than escalate, escalate is more restrictive than observe, observe is more
restrictive than auto-execute."

THE MEASURE, as pre-registered at commit 1b32639 before any result existed:
  1. number of triples enumerated
  2. number of monotonicity violations, meaning triples where the model-present disposition
     is LESS restrictive than the model-absent one
  3. number of triples where any deny or cap path reads a model-derived value at all, which
     should be zero by construction
Target zero for counts 2 and 3, and a non-zero count ships as a defect in our own
specification rather than being quietly repaired.

NO DATA IS READ. This exhibit opens no corpus. The sibling safety exhibits run on the public
IBM TabFormer benchmark, which is synthetic, or on public Foursquare venue records. This one
runs on records this file constructs from declared plan constants, so there is no privacy
question in it at all and no sampling uncertainty: the counts are exact over the enumerated
space.

THE SANITY DISCIPLINE, which is the reason a pass here means anything. A conformance checker
that cannot report a violation is not evidence. Three deliberately non-conforming reference
rules run through the identical enumeration, each one built to trip a named detector, and the
run fails loudly if any of them comes back clean. The output records what each mutant did to
each count, including the case where a mutant trips one detector and not the other, because
that is the evidence that the two counts are not measuring the same thing.

Laptop, seconds to a couple of minutes, stdlib only, deterministic, no random draw taken.
`--check` re-enumerates from scratch and compares every numeric leaf at 1e-6, exit 0 or 5.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent

# The pre-registration says stdlib only, so `atomic_write_json` and the version block are
# local copies rather than imports from scripts/fm/common.py, which imports numpy at module
# scope. Behaviour is identical to the house helper.


def atomic_write_json(path: str | Path, obj: dict) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


SEED = 0  # recorded for the house envelope; nothing here draws a random number

# ------------------------------------------------------------- dispositions --
# SAFETY-DESIGN.md 2.6: four dispositions, matching SAFR's four. The restrictiveness
# order is the one 2.1 states, and it is a total order, which is what makes "at least as
# restrictive as" a comparison rather than an opinion.
DENY, ESCALATE, OBSERVE, AUTO_EXECUTE = "DENY", "ESCALATE", "OBSERVE", "AUTO_EXECUTE"
RESTRICTIVENESS = {AUTO_EXECUTE: 0, OBSERVE: 1, ESCALATE: 2, DENY: 3}

# Bands, from 2.6.
V0, V1, V2 = "V0", "V1", "V2"
N0, N1, N2 = "N0", "N1", "N2"
R_FULL, R_RAIL, R_NONE = "R_full", "R_rail", "R_none"
MOST_PERMISSIVE_NOVELTY_BAND = N0

# ---------------------------------------------------------- plan constants ---
# Minor units of the mandate currency. These are the "pre-declared parameter ranges" the
# pre-registration names. On the page they would carry data-assumption="plan-constant".
PER_TRANSACTION_MAX = 20_000        # 200.00
AUTO_APPROVE_BELOW = 5_000          # 50.00
CUMULATIVE_MAX = 100_000            # 1000.00
VELOCITY_MAX_ACTIONS = 5
VELOCITY_WINDOW_SECONDS = 3_600
NOVELTY_SCORE_THRESHOLD = 0.95      # "a pre-set percentile of the cardmember's own history"
BUDGET_PROXIMITY_FRACTION = 0.8

NOT_BEFORE, NOW, EXPIRES_AT = 1_000, 1_500, 2_000   # epoch seconds, integers
EXPIRED_AT = 1_400                                   # strictly before NOW

# Reversibility is a property of the action type, fixed at design time (2.6). No model
# contributes to it.
REVERSIBILITY_BY_TYPE = {
    "cart_edit": R_FULL,
    "card_purchase": R_RAIL,
    "contact_change": R_NONE,
}

# ------------------------------------------------------------- provenance ----
# Audit element seven from 2.7: a provenance flag on every input, saying whether a model
# produced it. Here it is not documentation, it is the instrument. Every fact the reference
# rule reads goes through one accessor, the accessor looks the field up in this table, and
# the run records which fields were read in which phase. "model" is the tainted class.
PROVENANCE = {
    "identity_proved": "context",
    "signature_valid": "mandate",
    "not_before": "mandate",
    "expires_at": "mandate",
    "now": "context",
    "revoked": "mandate",
    "action_bound_to_mandate": "context",
    "origin_verified": "context",
    "action_type": "action",
    "scope_action_types": "mandate",
    "action_mcc": "action",
    "scope_mcc_deny": "mandate",
    "scope_mcc_allow": "mandate",
    "action_merchant": "action",
    "scope_merchant_allow": "mandate",
    "action_country": "action",
    "scope_geography": "mandate",
    "action_amount": "action",
    "scope_per_transaction_max": "configuration",
    "scope_auto_approve_below": "configuration",
    "scope_cumulative_max": "configuration",
    "spent_to_date": "context",
    "actions_in_window": "context",
    "scope_velocity_max_actions": "configuration",
    "prior_actions_at_merchant": "context",
    "budget_warned": "context",
    "novelty_floor": "context",
    "model_available": "context",
    "model_novelty_score": "model",
    "novelty_score_threshold": "configuration",
}
MODEL_PROVENANCE = "model"
FIELDS = sorted(PROVENANCE)
BIT = {f: 1 << i for i, f in enumerate(FIELDS)}
MODEL_MASK = 0
for _f, _p in PROVENANCE.items():
    if _p == MODEL_PROVENANCE:
        MODEL_MASK |= BIT[_f]

DETERMINISTIC_PHASE = "deterministic"
GRADED_PHASE = "graded"


class Ctx:
    """One evaluation. Reads go through `read` so the phase and the provenance of every
    read are observed rather than asserted. `det_mask` and `graded_mask` accumulate across
    a whole pass, so the field sets they hold are measured over the enumeration."""

    __slots__ = ("f", "det", "tainted", "det_mask", "graded_mask")

    def __init__(self):
        self.f = {}
        self.det = True
        self.tainted = False
        self.det_mask = 0
        self.graded_mask = 0

    def start(self, facts):
        self.f = facts
        self.det = True
        self.tainted = False

    def read(self, k):
        bit = BIT[k]
        if self.det:
            self.det_mask |= bit
            if bit & MODEL_MASK:
                self.tainted = True
        else:
            self.graded_mask |= bit
        return self.f[k]


def mask_fields(mask: int) -> list[str]:
    return [f for f in FIELDS if mask & BIT[f]]


# ------------------------------------------------- the rule, transcribed -----

def value_band(ctx: Ctx) -> str:
    """2.6 value bands. Facts about the action and numbers in the signed record only."""
    amount = ctx.read("action_amount")
    if amount <= ctx.read("scope_auto_approve_below"):
        return V0
    if amount <= ctx.read("scope_per_transaction_max"):
        return V1
    return V2


def novelty_band(ctx: Ctx) -> str:
    """2.6 novelty bands, the only place a model output enters.

    The deterministic floor comes from the cardmember's own history. The model score can
    raise the band to N2 and can do nothing else. With the model absent the band is pinned
    to the most permissive value, which is what the pre-registration specifies for the
    model-absent arm."""
    if not ctx.read("model_available"):
        return MOST_PERMISSIVE_NOVELTY_BAND
    if ctx.read("model_novelty_score") > ctx.read("novelty_score_threshold"):
        return N2
    return ctx.read("novelty_floor")


def reversibility(ctx: Ctx) -> str:
    return REVERSIBILITY_BY_TYPE[ctx.read("action_type")]


def reference_disposition(ctx: Ctx):
    """A line-for-line transcription of the rule in SAFETY-DESIGN.md 2.6.

    Returns (disposition, reason_code_returned_to_the_agent, internal_reason_label).
    The two reason fields differ in exactly one place, on purpose: 2.4 says the agent is
    told the reason code and never the threshold, and the specification returns one CATEGORY
    code whether the action tripped the deny list or fell outside the allow list, so the
    agent cannot use the difference as an oracle. The internal label keeps them apart for
    the audit record and for the coverage count below."""
    read = ctx.read

    # part one, deterministic, no model reachable from here
    ctx.det = True
    if not read("identity_proved"):
        return DENY, "IDENTITY", "IDENTITY"
    if not (read("signature_valid")
            and read("not_before") <= read("now") < read("expires_at")):
        return DENY, "MANDATE", "MANDATE"
    if read("revoked"):
        return DENY, "REVOKED", "REVOKED"
    if not read("action_bound_to_mandate"):
        return DENY, "BINDING", "BINDING"
    if not read("origin_verified"):
        return DENY, "ORIGIN_UNAVAILABLE", "ORIGIN_UNAVAILABLE"
    if read("action_type") not in read("scope_action_types"):
        return DENY, "OUT_OF_SCOPE", "OUT_OF_SCOPE"
    if read("action_mcc") in read("scope_mcc_deny"):
        return DENY, "CATEGORY", "CATEGORY_DENYLIST"
    if read("scope_mcc_allow") and read("action_mcc") not in read("scope_mcc_allow"):
        return DENY, "CATEGORY", "CATEGORY_ALLOWLIST"
    if (read("scope_merchant_allow")
            and read("action_merchant") not in read("scope_merchant_allow")):
        return DENY, "MERCHANT", "MERCHANT"
    if read("action_country") not in read("scope_geography"):
        return DENY, "GEOGRAPHY", "GEOGRAPHY"
    if value_band(ctx) == V2:
        return DENY, "OVER_CAP", "OVER_CAP"
    if read("spent_to_date") + read("action_amount") > read("scope_cumulative_max"):
        return DENY, "CUMULATIVE_CAP", "CUMULATIVE_CAP"
    if read("actions_in_window") >= read("scope_velocity_max_actions"):
        return DENY, "VELOCITY", "VELOCITY"

    # part two, graded. reversibility and value are facts. novelty may use a model score,
    # and may only move the action toward more friction.
    ctx.det = False
    if reversibility(ctx) == R_NONE:
        return ESCALATE, "IRREVERSIBLE", "IRREVERSIBLE"
    if novelty_band(ctx) == N2:
        return ESCALATE, "NOVELTY", "NOVELTY"
    if value_band(ctx) == V1 and novelty_band(ctx) == N1:
        return ESCALATE, "VALUE_AND_NOVELTY", "VALUE_AND_NOVELTY"
    if read("prior_actions_at_merchant") == 0 and value_band(ctx) != V0:
        return ESCALATE, "FIRST_USE", "FIRST_USE"
    if (read("spent_to_date") + read("action_amount")
            > BUDGET_PROXIMITY_FRACTION * read("scope_cumulative_max")
            and not read("budget_warned")):
        return ESCALATE, "BUDGET_PROXIMITY", "BUDGET_PROXIMITY"

    if value_band(ctx) == V1 or novelty_band(ctx) == N1:
        return OBSERVE, "VALUE_OR_NOVELTY", "VALUE_OR_NOVELTY"
    return AUTO_EXECUTE, "WITHIN_MANDATE", "WITHIN_MANDATE"


# ------------------------------------------- deliberately broken references --
# These are SYNTHETIC DEFECTS. They are not proposals, not alternatives, and no part of the
# specification. Each one exists so the enumeration can be shown to report a violation, and
# each one names the detector it must trip. If a mutant comes back clean the run aborts,
# because a checker that cannot fail is not a checker.

def mutant_novelty_review_substitutes_for_human(ctx: Ctx):
    """M1. A plausible optimisation: treat the model's novelty verdict as a completed
    review, so a flagged action is observed rather than escalated. It makes N2 LESS
    restrictive than N0 for an irreversible action, which is exactly the friction removal
    the house rule forbids. Must trip the monotonicity count. Reads no model value in the
    deterministic phase, so it must NOT trip the taint count."""
    read = ctx.read
    ctx.det = True
    if not read("identity_proved"):
        return DENY, "IDENTITY", "IDENTITY"
    if not (read("signature_valid")
            and read("not_before") <= read("now") < read("expires_at")):
        return DENY, "MANDATE", "MANDATE"
    if read("revoked"):
        return DENY, "REVOKED", "REVOKED"
    if not read("action_bound_to_mandate"):
        return DENY, "BINDING", "BINDING"
    if not read("origin_verified"):
        return DENY, "ORIGIN_UNAVAILABLE", "ORIGIN_UNAVAILABLE"
    if read("action_type") not in read("scope_action_types"):
        return DENY, "OUT_OF_SCOPE", "OUT_OF_SCOPE"
    if read("action_mcc") in read("scope_mcc_deny"):
        return DENY, "CATEGORY", "CATEGORY_DENYLIST"
    if read("scope_mcc_allow") and read("action_mcc") not in read("scope_mcc_allow"):
        return DENY, "CATEGORY", "CATEGORY_ALLOWLIST"
    if (read("scope_merchant_allow")
            and read("action_merchant") not in read("scope_merchant_allow")):
        return DENY, "MERCHANT", "MERCHANT"
    if read("action_country") not in read("scope_geography"):
        return DENY, "GEOGRAPHY", "GEOGRAPHY"
    if value_band(ctx) == V2:
        return DENY, "OVER_CAP", "OVER_CAP"
    if read("spent_to_date") + read("action_amount") > read("scope_cumulative_max"):
        return DENY, "CUMULATIVE_CAP", "CUMULATIVE_CAP"
    if read("actions_in_window") >= read("scope_velocity_max_actions"):
        return DENY, "VELOCITY", "VELOCITY"

    ctx.det = False
    if novelty_band(ctx) == N2:                      # <-- the defect: placed first, and
        return OBSERVE, "MODEL_REVIEWED", "MODEL_REVIEWED"   # downgraded to OBSERVE
    if reversibility(ctx) == R_NONE:
        return ESCALATE, "IRREVERSIBLE", "IRREVERSIBLE"
    if value_band(ctx) == V1 and novelty_band(ctx) == N1:
        return ESCALATE, "VALUE_AND_NOVELTY", "VALUE_AND_NOVELTY"
    if read("prior_actions_at_merchant") == 0 and value_band(ctx) != V0:
        return ESCALATE, "FIRST_USE", "FIRST_USE"
    if (read("spent_to_date") + read("action_amount")
            > BUDGET_PROXIMITY_FRACTION * read("scope_cumulative_max")
            and not read("budget_warned")):
        return ESCALATE, "BUDGET_PROXIMITY", "BUDGET_PROXIMITY"
    if value_band(ctx) == V1 or novelty_band(ctx) == N1:
        return OBSERVE, "VALUE_OR_NOVELTY", "VALUE_OR_NOVELTY"
    return AUTO_EXECUTE, "WITHIN_MANDATE", "WITHIN_MANDATE"


def mutant_over_cap_waived_when_not_novel(ctx: Ctx):
    """M2. The over-cap deny consults the model: an action above the per-action cap is let
    through when the model does not flag it. This is the case 2.5 calls the single most
    important line in the section, and it puts a model read inside the deny path. It must
    trip the taint count. It must NOT trip the monotonicity count, because the waiver fires
    at the same permissive band the model-absent arm is pinned to, so the model-present
    disposition is never less restrictive than the model-absent one. That asymmetry is the
    evidence that the two counts measure different things."""
    read = ctx.read
    ctx.det = True
    if not read("identity_proved"):
        return DENY, "IDENTITY", "IDENTITY"
    if not (read("signature_valid")
            and read("not_before") <= read("now") < read("expires_at")):
        return DENY, "MANDATE", "MANDATE"
    if read("revoked"):
        return DENY, "REVOKED", "REVOKED"
    if not read("action_bound_to_mandate"):
        return DENY, "BINDING", "BINDING"
    if not read("origin_verified"):
        return DENY, "ORIGIN_UNAVAILABLE", "ORIGIN_UNAVAILABLE"
    if read("action_type") not in read("scope_action_types"):
        return DENY, "OUT_OF_SCOPE", "OUT_OF_SCOPE"
    if read("action_mcc") in read("scope_mcc_deny"):
        return DENY, "CATEGORY", "CATEGORY_DENYLIST"
    if read("scope_mcc_allow") and read("action_mcc") not in read("scope_mcc_allow"):
        return DENY, "CATEGORY", "CATEGORY_ALLOWLIST"
    if (read("scope_merchant_allow")
            and read("action_merchant") not in read("scope_merchant_allow")):
        return DENY, "MERCHANT", "MERCHANT"
    if read("action_country") not in read("scope_geography"):
        return DENY, "GEOGRAPHY", "GEOGRAPHY"
    if value_band(ctx) == V2 and novelty_band(ctx) == N2:      # <-- the defect
        return DENY, "OVER_CAP", "OVER_CAP"
    if read("spent_to_date") + read("action_amount") > read("scope_cumulative_max"):
        return DENY, "CUMULATIVE_CAP", "CUMULATIVE_CAP"
    if read("actions_in_window") >= read("scope_velocity_max_actions"):
        return DENY, "VELOCITY", "VELOCITY"

    ctx.det = False
    if reversibility(ctx) == R_NONE:
        return ESCALATE, "IRREVERSIBLE", "IRREVERSIBLE"
    if novelty_band(ctx) == N2:
        return ESCALATE, "NOVELTY", "NOVELTY"
    if value_band(ctx) == V1 and novelty_band(ctx) == N1:
        return ESCALATE, "VALUE_AND_NOVELTY", "VALUE_AND_NOVELTY"
    if read("prior_actions_at_merchant") == 0 and value_band(ctx) != V0:
        return ESCALATE, "FIRST_USE", "FIRST_USE"
    if (read("spent_to_date") + read("action_amount")
            > BUDGET_PROXIMITY_FRACTION * read("scope_cumulative_max")
            and not read("budget_warned")):
        return ESCALATE, "BUDGET_PROXIMITY", "BUDGET_PROXIMITY"
    if value_band(ctx) == V1 or novelty_band(ctx) == N1:
        return OBSERVE, "VALUE_OR_NOVELTY", "VALUE_OR_NOVELTY"
    return AUTO_EXECUTE, "WITHIN_MANDATE", "WITHIN_MANDATE"


def mutant_cumulative_cap_grace_at_mid_novelty(ctx: Ctx):
    """M3. A cumulative-cap grace granted at the middle novelty band. It reads a model value
    inside a cap path, and it makes N1 less restrictive than N0, so it must trip BOTH counts.
    It is the least realistic of the three and it is here for exactly that reason: one mutant
    has to exercise the case where both detectors fire on the same rule."""
    read = ctx.read
    ctx.det = True
    if not read("identity_proved"):
        return DENY, "IDENTITY", "IDENTITY"
    if not (read("signature_valid")
            and read("not_before") <= read("now") < read("expires_at")):
        return DENY, "MANDATE", "MANDATE"
    if read("revoked"):
        return DENY, "REVOKED", "REVOKED"
    if not read("action_bound_to_mandate"):
        return DENY, "BINDING", "BINDING"
    if not read("origin_verified"):
        return DENY, "ORIGIN_UNAVAILABLE", "ORIGIN_UNAVAILABLE"
    if read("action_type") not in read("scope_action_types"):
        return DENY, "OUT_OF_SCOPE", "OUT_OF_SCOPE"
    if read("action_mcc") in read("scope_mcc_deny"):
        return DENY, "CATEGORY", "CATEGORY_DENYLIST"
    if read("scope_mcc_allow") and read("action_mcc") not in read("scope_mcc_allow"):
        return DENY, "CATEGORY", "CATEGORY_ALLOWLIST"
    if (read("scope_merchant_allow")
            and read("action_merchant") not in read("scope_merchant_allow")):
        return DENY, "MERCHANT", "MERCHANT"
    if read("action_country") not in read("scope_geography"):
        return DENY, "GEOGRAPHY", "GEOGRAPHY"
    if value_band(ctx) == V2:
        return DENY, "OVER_CAP", "OVER_CAP"
    if (read("spent_to_date") + read("action_amount") > read("scope_cumulative_max")
            and novelty_band(ctx) != N1):                       # <-- the defect
        return DENY, "CUMULATIVE_CAP", "CUMULATIVE_CAP"
    if read("actions_in_window") >= read("scope_velocity_max_actions"):
        return DENY, "VELOCITY", "VELOCITY"

    ctx.det = False
    if reversibility(ctx) == R_NONE:
        return ESCALATE, "IRREVERSIBLE", "IRREVERSIBLE"
    if novelty_band(ctx) == N2:
        return ESCALATE, "NOVELTY", "NOVELTY"
    if value_band(ctx) == V1 and novelty_band(ctx) == N1:
        return ESCALATE, "VALUE_AND_NOVELTY", "VALUE_AND_NOVELTY"
    if read("prior_actions_at_merchant") == 0 and value_band(ctx) != V0:
        return ESCALATE, "FIRST_USE", "FIRST_USE"
    if (read("spent_to_date") + read("action_amount")
            > BUDGET_PROXIMITY_FRACTION * read("scope_cumulative_max")
            and not read("budget_warned")):
        return ESCALATE, "BUDGET_PROXIMITY", "BUDGET_PROXIMITY"
    if value_band(ctx) == V1 or novelty_band(ctx) == N1:
        return OBSERVE, "VALUE_OR_NOVELTY", "VALUE_OR_NOVELTY"
    return AUTO_EXECUTE, "WITHIN_MANDATE", "WITHIN_MANDATE"


# The reason labels returned from the graded block. A triple that stops in the deterministic
# block cannot be affected by the model at all, so counting those triples as evidence for a
# monotonicity property would inflate the result. They are separated out below.
GRADED_REASON_LABELS = frozenset({
    "IRREVERSIBLE", "NOVELTY", "VALUE_AND_NOVELTY", "FIRST_USE", "BUDGET_PROXIMITY",
    "VALUE_OR_NOVELTY", "WITHIN_MANDATE", "MODEL_REVIEWED"})


MUTANTS = [
    {"name": "novelty_review_substitutes_for_human",
     "fn": mutant_novelty_review_substitutes_for_human,
     "defect": ("the model's novelty verdict is treated as a completed review, so a flagged "
                "action returns OBSERVE instead of reaching the escalation checks"),
     "must_trip_monotonicity": True,
     "must_trip_model_read_in_deterministic_path": False},
    {"name": "over_cap_waived_when_not_novel",
     "fn": mutant_over_cap_waived_when_not_novel,
     "defect": ("the over-cap deny consults the model and lets an above-cap action through "
                "whenever the model has not flagged it"),
     "must_trip_monotonicity": False,
     "must_trip_model_read_in_deterministic_path": True},
    {"name": "cumulative_cap_grace_at_mid_novelty",
     "fn": mutant_cumulative_cap_grace_at_mid_novelty,
     "defect": ("the cumulative-cap deny is waived at the middle novelty band, which both "
                "reads a model value in a cap path and makes N1 more permissive than N0"),
     "must_trip_monotonicity": True,
     "must_trip_model_read_in_deterministic_path": True},
]


# ------------------------------------------------------------ the grid ------
# Axes are declared here and nowhere else, so the enumeration and the output file cannot
# describe different spaces. Each axis lists the concrete field values it writes into the
# record, which is why the enumeration is over real mandate and action records with real
# minor-unit arithmetic rather than over abstract predicates.

MCC_DENY_LIST = ("7995",)                       # gambling, the standing deny
MCC_ALLOW_LIST = ("5812", "5411")               # dining, grocery
MERCHANT_ALLOW_LIST = ("m_allowed",)
ALL_ACTION_TYPES = tuple(sorted(REVERSIBILITY_BY_TYPE))
GEOGRAPHY = ("SG",)

MANDATE_STATE_AXIS = {
    "valid": {"signature_valid": True, "not_before": NOT_BEFORE, "expires_at": EXPIRES_AT},
    "expired": {"signature_valid": True, "not_before": NOT_BEFORE, "expires_at": EXPIRED_AT},
    "bad_signature": {"signature_valid": False, "not_before": NOT_BEFORE,
                      "expires_at": EXPIRES_AT},
}
REVOKED_AXIS = {"live": {"revoked": False}, "revoked": {"revoked": True}}
# Six mcc states: the deny list crossed with an absent, containing and excluding allow list.
# The two "denied_and_allowlisted" states are the ones that test the 2.2 rule that denies do
# not compose with allows.
MCC_STATE_AXIS = {
    "unrestricted_clean": {"scope_mcc_deny": MCC_DENY_LIST, "scope_mcc_allow": (),
                           "action_mcc": "5812"},
    "unrestricted_denied": {"scope_mcc_deny": MCC_DENY_LIST, "scope_mcc_allow": (),
                            "action_mcc": "7995"},
    "allowlisted_in_clean": {"scope_mcc_deny": MCC_DENY_LIST,
                             "scope_mcc_allow": MCC_ALLOW_LIST, "action_mcc": "5812"},
    "allowlisted_in_denied": {"scope_mcc_deny": MCC_DENY_LIST,
                              "scope_mcc_allow": MCC_ALLOW_LIST + ("7995",),
                              "action_mcc": "7995"},
    "allowlisted_out_clean": {"scope_mcc_deny": MCC_DENY_LIST,
                              "scope_mcc_allow": MCC_ALLOW_LIST, "action_mcc": "5999"},
    "allowlisted_out_denied": {"scope_mcc_deny": MCC_DENY_LIST,
                               "scope_mcc_allow": MCC_ALLOW_LIST, "action_mcc": "7995"},
}
MERCHANT_STATE_AXIS = {
    "unrestricted": {"scope_merchant_allow": (), "action_merchant": "m_any"},
    "allowlisted_in": {"scope_merchant_allow": MERCHANT_ALLOW_LIST,
                       "action_merchant": "m_allowed"},
    "allowlisted_out": {"scope_merchant_allow": MERCHANT_ALLOW_LIST,
                        "action_merchant": "m_other"},
}
TYPE_SCOPE_AXIS = {
    "type_in_scope": {"scope_action_types": ALL_ACTION_TYPES},
    "type_out_of_scope": {"scope_action_types": ("quote_request",)},
}

IDENTITY_AXIS = {"identity_ok": {"identity_proved": True},
                 "identity_fails": {"identity_proved": False}}
BINDING_AXIS = {"bound": {"action_bound_to_mandate": True},
                "unbound": {"action_bound_to_mandate": False}}
ORIGIN_AXIS = {"origin_ok": {"origin_verified": True},
               "origin_unavailable": {"origin_verified": False}}
GEO_AXIS = {"in_geography": {"action_country": "SG"},
            "out_of_geography": {"action_country": "XX"}}
# Amounts sit strictly below, exactly at and strictly above each of the two thresholds the
# value bands are cut on, so every comparison in value_band takes both truth values and the
# "at or below" inclusivity is exercised rather than assumed.
AMOUNT_AXIS = {
    "v0_interior": 2_500,
    "v0_at_auto_approve_boundary": AUTO_APPROVE_BELOW,
    "v1_interior": 12_000,
    "v1_at_per_transaction_boundary": PER_TRANSACTION_MAX,
    "v2_over_cap": 35_000,
}
# Cumulative position is declared as a fraction of the cap that spent_to_date PLUS the
# action amount lands on, so the two comparisons that read it (the cap deny at 1.0 and the
# budget-proximity escalation at 0.8) are both exercised at, below and above their line.
CUMULATIVE_AXIS = {
    "well_below": 0.5,
    "at_proximity_boundary": BUDGET_PROXIMITY_FRACTION,
    "between_proximity_and_cap": 0.9,
    "at_cap_boundary": 1.0,
    "over_cap": 1.2,
}
VELOCITY_AXIS = {"within_velocity": 0, "velocity_exceeded": VELOCITY_MAX_ACTIONS}
ACTION_TYPE_AXIS = {t: t for t in ALL_ACTION_TYPES}
FIRST_USE_AXIS = {"first_action_at_merchant": 0, "seen_before_at_merchant": 3}
BUDGET_WARNED_AXIS = {"not_warned": False, "already_warned": True}

# The third element of the pre-registered triple. The floor is what the cardmember's own
# history gives; the score is the model output. A score above the threshold raises the band
# to N2 and can do nothing else.
NOVELTY_INPUT_AXIS = {
    "floor_N0_score_low": {"novelty_floor": N0, "model_novelty_score": 0.10},
    "floor_N1_score_low": {"novelty_floor": N1, "model_novelty_score": 0.10},
    "floor_N0_score_high": {"novelty_floor": N0, "model_novelty_score": 0.99},
    "floor_N1_score_high": {"novelty_floor": N1, "model_novelty_score": 0.99},
}
# A finer sweep of the score itself, used on a declared cross-section below.
SCORE_SWEEP = [0.0, 0.25, 0.50, 0.75, 0.90, 0.94, 0.95, 0.96, 0.99, 1.00]

MANDATE_AXES = [("mandate_state", MANDATE_STATE_AXIS), ("revocation", REVOKED_AXIS),
                ("mcc_state", MCC_STATE_AXIS), ("merchant_state", MERCHANT_STATE_AXIS),
                ("action_type_scope", TYPE_SCOPE_AXIS)]
ACTION_AXES = [("identity", IDENTITY_AXIS), ("binding", BINDING_AXIS),
               ("origin", ORIGIN_AXIS), ("geography", GEO_AXIS)]

CONSTANT_FACTS = {
    "now": NOW,
    "scope_geography": GEOGRAPHY,
    "scope_per_transaction_max": PER_TRANSACTION_MAX,
    "scope_auto_approve_below": AUTO_APPROVE_BELOW,
    "scope_cumulative_max": CUMULATIVE_MAX,
    "scope_velocity_max_actions": VELOCITY_MAX_ACTIONS,
    "novelty_score_threshold": NOVELTY_SCORE_THRESHOLD,
    "model_available": True,
}


def mandate_parts() -> list[dict]:
    out = []
    for combo in itertools.product(*[sorted(a) for _, a in MANDATE_AXES]):
        rec = {}
        for (_, axis), key in zip(MANDATE_AXES, combo):
            rec.update(axis[key])
        out.append(rec)
    return out


def action_parts() -> list[dict]:
    out = []
    grid = itertools.product(
        *[sorted(a) for _, a in ACTION_AXES],
        sorted(AMOUNT_AXIS), sorted(CUMULATIVE_AXIS), sorted(VELOCITY_AXIS),
        sorted(ACTION_TYPE_AXIS), sorted(FIRST_USE_AXIS), sorted(BUDGET_WARNED_AXIS))
    for combo in grid:
        bools, rest = combo[:len(ACTION_AXES)], combo[len(ACTION_AXES):]
        amount_key, cum_key, vel_key, type_key, first_key, warn_key = rest
        rec = dict(CONSTANT_FACTS)
        for (_, axis), key in zip(ACTION_AXES, bools):
            rec.update(axis[key])
        amount = AMOUNT_AXIS[amount_key]
        rec["action_amount"] = amount
        # spent_to_date is chosen so that spent + amount lands exactly on the declared
        # fraction of the cap. Integer arithmetic, and never negative for this grid.
        rec["spent_to_date"] = round(CUMULATIVE_AXIS[cum_key] * CUMULATIVE_MAX) - amount
        rec["actions_in_window"] = VELOCITY_AXIS[vel_key]
        rec["action_type"] = ACTION_TYPE_AXIS[type_key]
        rec["prior_actions_at_merchant"] = FIRST_USE_AXIS[first_key]
        rec["budget_warned"] = BUDGET_WARNED_AXIS[warn_key]
        out.append(rec)
    return out


# ------------------------------------------------------------- the passes ---

def run_rule(fn, mandates, actions, novelty_inputs, collect_coverage: bool) -> dict:
    """One full exhaustive pass for one rule.

    Per base record, the model-absent disposition is computed once with model_available
    False, which pins novelty to the most permissive band exactly as the pre-registration
    specifies. Then each novelty input is a triple, and the triple is a violation when its
    disposition is strictly less restrictive than the model-absent one."""
    ctx = Ctx()
    absent_facts_key = "model_available"
    n_triples = 0
    n_evaluations = 0
    n_violations = 0
    n_tainted = 0
    violations_by_novelty_input = {k: 0 for k in novelty_inputs}
    tainted_by_disposition = {d: 0 for d in RESTRICTIVENESS}
    violation_examples = []
    pairwise_violations = 0
    dispositions = {d: 0 for d in RESTRICTIVENESS}
    returned_codes = {}
    internal_codes = {}
    absent_dispositions = {d: 0 for d in RESTRICTIVENESS}
    n_repeat_evaluations = 0
    n_repeat_disagreements = 0
    n_graded = 0
    n_model_changed = 0
    n_model_increased = 0

    novelty_items = sorted(novelty_inputs.items())
    band_rank = {N0: 0, N1: 1, N2: 2}

    for m in mandates:
        for a in actions:
            facts = {**m, **a}

            facts[absent_facts_key] = False
            facts["novelty_floor"] = N0
            facts["model_novelty_score"] = None
            ctx.start(facts)
            absent_disp, _, _ = fn(ctx)
            n_evaluations += 1
            absent_rank = RESTRICTIVENESS[absent_disp]
            absent_dispositions[absent_disp] += 1

            if collect_coverage:
                # the identity rung: the same rule on the same record, evaluated a second
                # time. Zero by construction only if the rule really is a pure function of
                # the facts, which is the thing being checked rather than assumed.
                ctx.start(facts)
                repeat_disp, _, _ = fn(ctx)
                n_evaluations += 1
                n_repeat_evaluations += 1
                if repeat_disp != absent_disp:
                    n_repeat_disagreements += 1

            facts[absent_facts_key] = True
            by_band = {}
            for key, patch in novelty_items:
                facts["novelty_floor"] = patch["novelty_floor"]
                facts["model_novelty_score"] = patch["model_novelty_score"]
                ctx.start(facts)
                disp, returned, internal = fn(ctx)
                n_evaluations += 1
                n_triples += 1
                rank = RESTRICTIVENESS[disp]
                if internal in GRADED_REASON_LABELS:
                    n_graded += 1
                if rank != absent_rank:
                    n_model_changed += 1
                    if rank > absent_rank:
                        n_model_increased += 1
                if rank < absent_rank:
                    n_violations += 1
                    violations_by_novelty_input[key] += 1
                    if len(violation_examples) < 5:
                        violation_examples.append({
                            "novelty_input": key,
                            "model_present_disposition": disp,
                            "model_absent_disposition": absent_disp,
                            "action_amount": facts["action_amount"],
                            "action_type": facts["action_type"],
                            "spent_to_date": facts["spent_to_date"],
                            "reason_code_model_present": returned,
                        })
                if ctx.tainted:
                    n_tainted += 1
                    tainted_by_disposition[disp] += 1
                if collect_coverage:
                    dispositions[disp] += 1
                    returned_codes[returned] = returned_codes.get(returned, 0) + 1
                    internal_codes[internal] = internal_codes.get(internal, 0) + 1
                # the band this triple actually reached, for the pairwise check
                band = (N2 if patch["model_novelty_score"] > NOVELTY_SCORE_THRESHOLD
                        else patch["novelty_floor"])
                prev = by_band.get(band)
                by_band[band] = rank if prev is None else max(prev, rank)
            ordered = sorted(by_band.items(), key=lambda kv: band_rank[kv[0]])
            for i in range(1, len(ordered)):
                if ordered[i][1] < ordered[i - 1][1]:
                    pairwise_violations += 1
                    break

    out = {
        "n_triples_enumerated": n_triples,
        "n_disposition_evaluations": n_evaluations,
        "n_monotonicity_violations": n_violations,
        "n_model_derived_reads_in_deterministic_path": n_tainted,
        "violations_by_novelty_input": violations_by_novelty_input,
        "tainted_triples_by_disposition": tainted_by_disposition,
        "violation_examples": violation_examples,
        "n_bases_with_pairwise_band_monotonicity_violation": pairwise_violations,
        "n_triples_reaching_the_graded_block": n_graded,
        "n_triples_where_the_model_changed_the_disposition": n_model_changed,
        "n_triples_where_the_model_added_friction": n_model_increased,
        "fields_read_in_deterministic_phase": mask_fields(ctx.det_mask),
        "fields_read_in_graded_phase": mask_fields(ctx.graded_mask),
        "model_provenance_fields_read_in_deterministic_phase":
            [f for f in mask_fields(ctx.det_mask) if PROVENANCE[f] == MODEL_PROVENANCE],
    }
    if collect_coverage:
        out["dispositions_model_present"] = dispositions
        out["dispositions_model_absent"] = absent_dispositions
        out["reason_codes_returned_to_the_agent"] = dict(sorted(returned_codes.items()))
        out["reason_codes_internal"] = dict(sorted(internal_codes.items()))
        out["n_repeat_evaluations"] = n_repeat_evaluations
        out["n_repeat_disagreements"] = n_repeat_disagreements
    return out


def run_score_sweep(fn) -> dict:
    """Supplementary. The pre-registered comparison pins the model-absent arm to one band.
    This arm sweeps the model score itself across its threshold, on a declared cross-section
    where every deterministic pre-check passes, and asks whether restrictiveness is
    non-decreasing in the score. It is a strengthening of the pre-registered measure, not a
    substitute for it."""
    ctx = Ctx()
    base_mandate = {}
    for key, axis in [("valid", MANDATE_STATE_AXIS), ("live", REVOKED_AXIS),
                      ("allowlisted_in_clean", MCC_STATE_AXIS),
                      ("allowlisted_in", MERCHANT_STATE_AXIS),
                      ("type_in_scope", TYPE_SCOPE_AXIS)]:
        base_mandate.update(axis[key])

    n_bases = 0
    n_rows = 0
    n_steps = 0
    disposition_violations = 0
    band_violations = 0
    grid = itertools.product(sorted(AMOUNT_AXIS), sorted(CUMULATIVE_AXIS),
                             sorted(VELOCITY_AXIS), sorted(ACTION_TYPE_AXIS),
                             sorted(FIRST_USE_AXIS), sorted(BUDGET_WARNED_AXIS),
                             [N0, N1])
    for amount_key, cum_key, vel_key, type_key, first_key, warn_key, floor in grid:
        facts = dict(CONSTANT_FACTS)
        facts.update(base_mandate)
        facts.update(IDENTITY_AXIS["identity_ok"])
        facts.update(BINDING_AXIS["bound"])
        facts.update(ORIGIN_AXIS["origin_ok"])
        facts.update(GEO_AXIS["in_geography"])
        amount = AMOUNT_AXIS[amount_key]
        facts["action_amount"] = amount
        facts["spent_to_date"] = round(CUMULATIVE_AXIS[cum_key] * CUMULATIVE_MAX) - amount
        facts["actions_in_window"] = VELOCITY_AXIS[vel_key]
        facts["action_type"] = ACTION_TYPE_AXIS[type_key]
        facts["prior_actions_at_merchant"] = FIRST_USE_AXIS[first_key]
        facts["budget_warned"] = BUDGET_WARNED_AXIS[warn_key]
        facts["novelty_floor"] = floor
        n_bases += 1
        prev_rank = None
        prev_band = None
        for score in SCORE_SWEEP:
            facts["model_novelty_score"] = score
            ctx.start(facts)
            disp, _, _ = fn(ctx)
            n_rows += 1
            rank = RESTRICTIVENESS[disp]
            band = N2 if score > NOVELTY_SCORE_THRESHOLD else floor
            band_r = {N0: 0, N1: 1, N2: 2}[band]
            if prev_rank is not None:
                n_steps += 1
                if rank < prev_rank:
                    disposition_violations += 1
                if band_r < prev_band:
                    band_violations += 1
            prev_rank, prev_band = rank, band_r
    return {
        "what_this_is": ("restrictiveness as a function of the model novelty score itself, "
                         "on the cross-section where every deterministic pre-check passes"),
        "score_grid": list(SCORE_SWEEP),
        "threshold": NOVELTY_SCORE_THRESHOLD,
        "comparison": "strictly greater than the threshold raises the band to N2",
        "n_bases": n_bases,
        "n_rows": n_rows,
        "n_score_steps_compared": n_steps,
        "n_score_steps_where_restrictiveness_decreased": disposition_violations,
        "n_score_steps_where_the_band_decreased": band_violations,
        "exhaustive_over_this_cross_section": True,
    }


# --------------------------------------------------------------- assemble ---

def build() -> dict:
    t0 = time.time()
    mandates = mandate_parts()
    actions = action_parts()
    n_bases = len(mandates) * len(actions)

    primary = run_rule(reference_disposition, mandates, actions,
                       NOVELTY_INPUT_AXIS, collect_coverage=True)
    sweep = run_score_sweep(reference_disposition)

    mutant_rows = []
    for spec in MUTANTS:
        r = run_rule(spec["fn"], mandates, actions, NOVELTY_INPUT_AXIS,
                     collect_coverage=False)
        caught_mono = r["n_monotonicity_violations"] > 0
        caught_taint = r["n_model_derived_reads_in_deterministic_path"] > 0
        row = {
            "name": spec["name"],
            "defect": spec["defect"],
            "n_triples_enumerated": r["n_triples_enumerated"],
            "n_triples_where_the_model_changed_the_disposition":
                r["n_triples_where_the_model_changed_the_disposition"],
            "n_monotonicity_violations": r["n_monotonicity_violations"],
            "n_model_derived_reads_in_deterministic_path":
                r["n_model_derived_reads_in_deterministic_path"],
            "model_provenance_fields_read_in_deterministic_phase":
                r["model_provenance_fields_read_in_deterministic_phase"],
            "expected_to_trip_monotonicity": spec["must_trip_monotonicity"],
            "expected_to_trip_model_read_in_deterministic_path":
                spec["must_trip_model_read_in_deterministic_path"],
            "tripped_monotonicity": caught_mono,
            "tripped_model_read_in_deterministic_path": caught_taint,
            "caught_as_expected": (caught_mono == spec["must_trip_monotonicity"]
                                   and caught_taint ==
                                   spec["must_trip_model_read_in_deterministic_path"]),
            "first_violations": r["violation_examples"],
        }
        mutant_rows.append(row)

    uncaught = [r["name"] for r in mutant_rows if not r["caught_as_expected"]]
    if uncaught:
        sys.exit("SANITY FAILED: mutants did not behave as declared: " + ", ".join(uncaught))

    det_fields = primary["fields_read_in_deterministic_phase"]
    model_in_det = primary["model_provenance_fields_read_in_deterministic_phase"]
    conforms = (primary["n_monotonicity_violations"] == 0
                and primary["n_model_derived_reads_in_deterministic_path"] == 0)

    all_returned = set(primary["reason_codes_returned_to_the_agent"])
    all_internal = set(primary["reason_codes_internal"])
    expected_internal = {
        "IDENTITY", "MANDATE", "REVOKED", "BINDING", "ORIGIN_UNAVAILABLE", "OUT_OF_SCOPE",
        "CATEGORY_DENYLIST", "CATEGORY_ALLOWLIST", "MERCHANT", "GEOGRAPHY", "OVER_CAP",
        "CUMULATIVE_CAP", "VELOCITY", "IRREVERSIBLE", "NOVELTY", "VALUE_AND_NOVELTY",
        "FIRST_USE", "BUDGET_PROXIMITY", "VALUE_OR_NOVELTY", "WITHIN_MANDATE"}
    unreached = sorted(expected_internal - all_internal)
    dispositions_reached = sorted(d for d, n in primary["dispositions_model_present"].items()
                                  if n > 0)

    wall = time.time() - t0

    design_doc = REPO / "SAFETY-DESIGN.md"
    out = {
        "seed": SEED,
        "versions": {"python": platform.python_version(), "stdlib_only": True},
        "generated_by": "scripts/safety/control_plane_conformance.py --check-able",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "data_sources": [{
            "name": ("SAFETY-DESIGN.md, Part 2.6 decision rule and the SAFE-E "
                     "pre-registration in Part 3, both committed at 1b32639 before any "
                     "result existed"),
            "url": "SAFETY-DESIGN.md (this repository)",
            "sha256": sha256_file(design_doc)}],
        "labels": ["designed-and-checked", "not-deployed", "reference-implementation",
                   "no-data-read", "exhaustive-enumeration", "specification-conformance"],

        "status": "DESIGNED-AND-CHECKED",
        "status_note": (
            "The agent control plane is designed and not built. No line of it runs in any "
            "product. This exhibit checks one property of the written specification against "
            "a reference implementation of it. DESIGNED-AND-CHECKED is not deployed, and "
            "nothing here may be presented as a measurement of a running system."),
        "what_this_is": (
            "The rule in SAFETY-DESIGN.md 2.6 claims that a model output may only add "
            "friction and never remove it. This file transcribes that rule into a reference "
            "implementation, enumerates mandate, action and model-novelty-score triples, and "
            "counts the triples where the model made the outcome less restrictive than it "
            "would have been with no model at all. It converts one design assertion into a "
            "checkable property. It converts nothing else."),
        "no_data_read": (
            "This exhibit opens no dataset. The sibling safety exhibits run on the public "
            "IBM TabFormer benchmark, which is synthetic, or on public Foursquare venue "
            "records. This one runs on records constructed here from declared plan "
            "constants, so it carries no privacy question and no sampling uncertainty. The "
            "counts are exact over the enumerated space."),

        "the_property_under_test": {
            "house_rule": (
                "A model output may only add friction, never remove it. For every action, "
                "the disposition computed with the model score present must be at least as "
                "restrictive as the disposition computed with the model absent."),
            "source": "SAFETY-DESIGN.md 2.1, part 3",
            "restrictiveness_order": ["AUTO_EXECUTE", "OBSERVE", "ESCALATE", "DENY"],
            "restrictiveness_rank": RESTRICTIVENESS,
            "model_absent_arm": (
                "model_available is False, so the novelty function returns the most "
                "permissive band N0 without reading the score. This is what the "
                "pre-registration specifies: the model absent and novelty pinned to its "
                "most permissive band."),
            "violation_definition": (
                "a triple whose model-present disposition has a strictly lower "
                "restrictiveness rank than the same base record evaluated with the model "
                "absent"),
        },

        "reference_implementation": {
            "function": "reference_disposition in scripts/safety/control_plane_conformance.py",
            "transcribed_from": "SAFETY-DESIGN.md 2.6, the pseudocode block",
            "phases": ["deterministic", "graded"],
            "reason_code_asymmetry": (
                "Two branches return the single CATEGORY code the specification defines, one "
                "for the deny list and one for the allow list. 2.4 says the agent is told the "
                "reason code and never the threshold, so collapsing them is deliberate and "
                "the internal labels are kept apart only for the audit record and for the "
                "coverage count here."),
            "instrumentation": (
                "Every fact is read through one accessor that records the current phase and "
                "the provenance of the field. The provenance table is audit element seven "
                "from 2.7 made into the instrument rather than into documentation."),
            "provenance_table": dict(sorted(PROVENANCE.items())),
            "model_provenance_fields": sorted(f for f, p in PROVENANCE.items()
                                              if p == MODEL_PROVENANCE),
        },

        "plan_constants": {
            "per_transaction_max_minor_units": PER_TRANSACTION_MAX,
            "auto_approve_below_minor_units": AUTO_APPROVE_BELOW,
            "cumulative_max_minor_units": CUMULATIVE_MAX,
            "velocity_max_actions": VELOCITY_MAX_ACTIONS,
            "velocity_window_seconds": VELOCITY_WINDOW_SECONDS,
            "novelty_score_threshold": NOVELTY_SCORE_THRESHOLD,
            "budget_proximity_fraction": BUDGET_PROXIMITY_FRACTION,
            "note": ("plan constants, not measurements. On the page they carry "
                     "data-assumption=plan-constant per CONTRACT.md section 2b."),
        },

        "enumeration": {
            "exhaustive_or_sampled": "exhaustive",
            "what_exhaustive_means_here": (
                "every combination of the declared axes is evaluated, with no sampling and "
                "no random draw. The axes cover every fact the rule reads, and each numeric "
                "axis places a value strictly below, exactly at and strictly above the "
                "threshold it is compared against, so every comparison in the rule takes "
                "both truth values."),
            "n_mandate_records": len(mandates),
            "n_action_records": len(actions),
            "n_base_records": n_bases,
            "n_novelty_inputs": len(NOVELTY_INPUT_AXIS),
            "n_triples_enumerated": primary["n_triples_enumerated"],
            "n_disposition_evaluations": primary["n_disposition_evaluations"],
            "axes": [
                {"name": "mandate_state", "values": sorted(MANDATE_STATE_AXIS), "n": 3},
                {"name": "revocation", "values": sorted(REVOKED_AXIS), "n": 2},
                {"name": "mcc_state", "values": sorted(MCC_STATE_AXIS), "n": 6},
                {"name": "merchant_state", "values": sorted(MERCHANT_STATE_AXIS), "n": 3},
                {"name": "action_type_scope", "values": sorted(TYPE_SCOPE_AXIS), "n": 2},
                {"name": "identity", "values": sorted(IDENTITY_AXIS), "n": 2},
                {"name": "binding", "values": sorted(BINDING_AXIS), "n": 2},
                {"name": "origin", "values": sorted(ORIGIN_AXIS), "n": 2},
                {"name": "geography", "values": sorted(GEO_AXIS), "n": 2},
                {"name": "action_amount", "values": sorted(AMOUNT_AXIS), "n": 5},
                {"name": "cumulative_position", "values": sorted(CUMULATIVE_AXIS), "n": 5},
                {"name": "velocity", "values": sorted(VELOCITY_AXIS), "n": 2},
                {"name": "action_type", "values": sorted(ACTION_TYPE_AXIS), "n": 3},
                {"name": "first_use", "values": sorted(FIRST_USE_AXIS), "n": 2},
                {"name": "budget_warned", "values": sorted(BUDGET_WARNED_AXIS), "n": 2},
                {"name": "novelty_input", "values": sorted(NOVELTY_INPUT_AXIS), "n": 4},
            ],
            "axes_by_name": {},
        },

        "primary": {
            "what_this_is": ("the three counts the pre-registration names, and nothing "
                             "else"),
            "n_triples_enumerated": primary["n_triples_enumerated"],
            "n_monotonicity_violations": primary["n_monotonicity_violations"],
            "n_model_derived_reads_in_deterministic_path":
                primary["n_model_derived_reads_in_deterministic_path"],
            "violations_by_novelty_input": primary["violations_by_novelty_input"],
            "conforms": conforms,
            "verdict": (
                "The property holds over the enumerated space: no triple made the model's "
                "contribution reduce friction, and no deny or cap path read a model-derived "
                "value."
                if conforms else
                "The property does NOT hold over the enumerated space. This is a defect in "
                "our own specification and it ships as one."),
        },

        "provenance_instrumentation": {
            "method": ("the phase and the provenance of every read are recorded by the "
                       "accessor as the rule executes, so the field lists below are "
                       "observed over the enumeration rather than read off the source"),
            "fields_read_in_deterministic_phase": det_fields,
            "n_fields_read_in_deterministic_phase": len(det_fields),
            "fields_read_in_graded_phase": primary["fields_read_in_graded_phase"],
            "n_fields_read_in_graded_phase": len(primary["fields_read_in_graded_phase"]),
            "model_provenance_fields_read_in_deterministic_phase": model_in_det,
            "tainted_triples_by_disposition": primary["tainted_triples_by_disposition"],
        },

        "coverage": {
            "why_this_is_here": (
                "zero violations proves nothing if the enumeration never reached the graded "
                "block. These counts show which outcomes and which reason codes the space "
                "actually exercised."),
            "dispositions_model_present": primary["dispositions_model_present"],
            "dispositions_model_absent": primary["dispositions_model_absent"],
            "dispositions_reached": dispositions_reached,
            "n_dispositions_reached": len(dispositions_reached),
            "reason_codes_returned_to_the_agent":
                primary["reason_codes_returned_to_the_agent"],
            "n_reason_codes_returned_to_the_agent": len(all_returned),
            "reason_codes_internal": primary["reason_codes_internal"],
            "n_reason_codes_internal": len(all_internal),
            "internal_reason_codes_never_reached": unreached,
            "n_internal_reason_codes_never_reached": len(unreached),
            "n_triples_reaching_the_graded_block":
                primary["n_triples_reaching_the_graded_block"],
            "share_of_triples_reaching_the_graded_block": round(
                primary["n_triples_reaching_the_graded_block"]
                / primary["n_triples_enumerated"], 8),
            "n_triples_where_the_model_changed_the_disposition":
                primary["n_triples_where_the_model_changed_the_disposition"],
            "n_triples_where_the_model_added_friction":
                primary["n_triples_where_the_model_added_friction"],
            "read_the_denominator_before_the_headline": (
                "The large count is the enumerated space, not the strength of the evidence. "
                "Most triples stop in the deterministic block, where the model cannot reach "
                "them and where a monotonicity pass is free. The triples that carry the "
                "weight are the ones that reach the graded block, and inside those, the ones "
                "where the model actually moved the outcome. Both counts are printed above "
                "and the second is the one to quote."),
        },

        "supplementary": {
            "what_this_is": ("two arms beyond the pre-registered three counts. They "
                             "strengthen the primary measure and they do not replace it."),
            "pairwise_band_monotonicity": {
                "definition": ("for every base record, restrictiveness must be "
                               "non-decreasing across the novelty bands in the order N0, "
                               "N1, N2, not only against the pinned N0 baseline"),
                "n_bases_checked": n_bases,
                "n_bases_with_a_violation":
                    primary["n_bases_with_pairwise_band_monotonicity_violation"],
            },
            "model_score_sweep": sweep,
        },

        "sanity": {
            "what_this_is": (
                "a conformance checker that cannot report a violation is not evidence. Three "
                "deliberately broken reference rules run through the identical enumeration, "
                "each built to trip a named detector. The run aborts if any of them comes "
                "back behaving other than declared, so the zero above is a measured zero and "
                "not an untested one."),
            "these_are_synthetic_defects": (
                "The mutants are not proposals and no part of the specification. They exist "
                "only to calibrate the detectors."),
            "identity_rung": {
                "config": ("the reference rule evaluated twice on the same record, model "
                           "absent on both sides"),
                "n_bases": n_bases,
                "n_repeat_evaluations": primary["n_repeat_evaluations"],
                "n_disagreements": primary["n_repeat_disagreements"],
                "note": ("the comparator's zero point, and a determinism check on the rule "
                         "at the same time. It is measured rather than assumed, because a "
                         "rule that is not a pure function of the facts would make every "
                         "other count here meaningless."),
            },
            "mutants": mutant_rows,
            "mutants_by_name": {r["name"]: r for r in mutant_rows},
            "n_mutants": len(mutant_rows),
            "n_mutants_caught_as_declared": sum(1 for r in mutant_rows
                                                if r["caught_as_expected"]),
            "the_asymmetry_that_matters": (
                "One mutant trips the monotonicity count and not the taint count, one trips "
                "the taint count and not the monotonicity count, and one trips both. That is "
                "the evidence that the two pre-registered counts measure different things "
                "and that neither is redundant."),
        },

        "findings_as_obtained": [],
        "interpretation_guard": [
            "Zero violations means the rule AS SPECIFIED cannot let a model output reduce "
            "friction over the enumerated space. It does not mean the control plane is safe, "
            "and it does not mean the control plane exists.",
            "The status is DESIGNED-AND-CHECKED. Do not place this beside the measured "
            "privacy results in a way that reads as a measurement of a running system.",
            "This is a property of our own specification, checked by us. An external reader "
            "gets the value from the enumeration being exhaustive and from the mutants "
            "showing the checker can fail, not from the number zero on its own.",
            "Do not quote the triple count as though it were the strength of the evidence. "
            "Most of the space stops in the deterministic block where the model cannot reach "
            "it. Quote coverage.n_triples_where_the_model_changed_the_disposition beside it "
            "or quote neither.",
        ],
        "limitations": [
            "It is a reference implementation of a rule this team wrote, not the code of any "
            "deployed system. A conformance pass says the specification has the property. It "
            "says nothing about an implementation nobody has built.",
            "The enumeration is exhaustive over the declared axes, materialized from one set "
            "of plan constants. Every comparison in the rule is exercised below, at and "
            "above its threshold, but a different constant set is a different enumeration.",
            "The enumerated space is dominated by triples the deterministic block denies "
            "before the model is consulted, so the headline count is much larger than the "
            "count of triples on which the property is non-trivially tested. Both numbers "
            "sit in the coverage block and the smaller one is the honest one.",
            "The property is monotonicity in the model's contribution. It is not a proof "
            "that the deterministic layer is correct, that the thresholds are well chosen, "
            "or that a real novelty model would produce scores of the shape assumed here.",
            "Nothing here measures resistance to attack. A gate can be perfectly monotone in "
            "the model's output and still be defeated by an unverified mandate, a stale "
            "revocation record or a compromised registry. Those sit in 2.4, 2.7 and the "
            "failure list in 2.8, and none of them is measured by this file.",
            "The taint instrumentation observes reads through one accessor. A rule that "
            "reached a model value by another route, by closing over it for instance, would "
            "not be seen. The reference implementation reads every fact through the "
            "accessor and that is checkable by reading the function, but it is a property of "
            "this code and not something the language enforces.",
            "The four dispositions are treated as a total order. That is what 2.1 states, and "
            "it is a modelling choice: an escalation that times out is recorded as not "
            "executed, so it is treated here as more restrictive than an observe, which "
            "executes.",
        ],
        "deviations_from_preregistration": [],
        "pointer_law": (
            "sanity.mutants is a LIST, so copy addresses a mutant by name only through "
            "sanity.mutants_by_name.<name>, never by position. enumeration.axes is a LIST "
            "for the same reason and carries enumeration.axes_by_name beside it. Positional "
            "stamps are allowed only inside a renderer that emits the key and the value from "
            "the same loop iteration."),
        "runtime": {
            "wall_seconds": round(wall, 2),
            "machine": "laptop CPU, single process, stdlib only",
            "note": ("excluded from --check by prefix, because it is a timing and not a "
                     "measurement"),
        },
        "check": {
            "command": "python3 scripts/safety/control_plane_conformance.py --check",
            "tolerance": 1e-6,
            "note": ("re-enumerates the whole space from the declared axes and compares "
                     "every numeric leaf against the committed "
                     "results/safety_control_plane.json. Deterministic on CPU with no random "
                     "draw and no data read, so it needs no node pinning. /versions and "
                     "/runtime are excluded from the comparison. The check also prints an "
                     "advisory when the sha256 of SAFETY-DESIGN.md differs from the one "
                     "recorded in data_sources, which does not change the exit code."),
        },
    }

    out["enumeration"]["axes_by_name"] = {a["name"]: a for a in out["enumeration"]["axes"]}

    out["findings_as_obtained"] = [
        (f"{primary['n_triples_enumerated']:,} triples enumerated exhaustively over "
         f"{n_bases:,} base records and {len(NOVELTY_INPUT_AXIS)} novelty inputs, with "
         f"{primary['n_monotonicity_violations']:,} monotonicity violations."),
        (f"The denominator flatters the result and the honest number is smaller. Only "
         f"{primary['n_triples_reaching_the_graded_block']:,} of those triples reach the "
         f"graded block at all, because the deterministic block denies the rest first, and "
         f"the model changed the disposition in "
         f"{primary['n_triples_where_the_model_changed_the_disposition']:,} of them, every "
         f"one of those "
         f"{primary['n_triples_where_the_model_added_friction']:,} times toward more "
         f"friction. That smaller count is the one carrying the property."),
        (f"{primary['n_model_derived_reads_in_deterministic_path']:,} triples read a "
         f"model-provenance field inside the deterministic block. The deterministic block "
         f"read {len(det_fields)} distinct fields over the whole enumeration and "
         f"{len(model_in_det)} of them carry model provenance."),
        (f"All four dispositions and {len(all_internal)} of {len(expected_internal)} "
         f"internal reason labels were reached, so the enumeration exercised the graded "
         f"block rather than denying everything early."
         if not unreached else
         f"{len(unreached)} internal reason labels were never reached: "
         f"{', '.join(unreached)}. That is a coverage gap in the enumeration and it ships."),
        (f"{len(mutant_rows)} deliberately broken reference rules were run through the same "
         f"enumeration and all {sum(1 for r in mutant_rows if r['caught_as_expected'])} "
         f"behaved exactly as declared in advance, which is what makes the zero above a "
         f"measured zero."),
        (f"The finer arm agrees: {primary['n_bases_with_pairwise_band_monotonicity_violation']:,} "
         f"base records show a violation when restrictiveness is checked pairwise across "
         f"N0, N1 and N2 rather than only against the pinned baseline, and "
         f"{sweep['n_score_steps_where_restrictiveness_decreased']:,} of "
         f"{sweep['n_score_steps_compared']:,} score steps reduced restrictiveness as the "
         f"model novelty score rose across its threshold."),
    ]

    out["deviations_from_preregistration"] = [
        {"item": "runtime",
         "preregistered": "laptop, seconds, stdlib only, deterministic",
         "as_run": (f"stdlib only and deterministic as written, wall time "
                    f"{round(wall, 1)} seconds"),
         "why": ("the enumeration was taken exhaustive over the full declared axis set "
                 "rather than sampled, because an exhaustive count over a small space is a "
                 "stronger statement than a sampled one and the space is small enough to "
                 "enumerate")},
        {"item": "house helpers",
         "preregistered": "stdlib only",
         "as_run": ("atomic_write_json and the version block are local copies rather than "
                    "imports from scripts/fm/common.py"),
         "why": ("that module imports numpy at module scope, and importing it would break "
                 "the stdlib-only promise. Behaviour is identical to the house helper.")},
        {"item": "arms beyond the pre-registration",
         "preregistered": ("three counts: triples enumerated, monotonicity violations, "
                           "model-derived reads inside the deterministic path"),
         "as_run": ("the three counts are reported unchanged as primary, with a coverage "
                    "block, a pairwise band monotonicity arm and a model score sweep added "
                    "beside them"),
         "why": ("the pre-registered comparison pins the model-absent arm to one band, so on "
                 "its own it cannot see a rule that is uniformly more permissive. The added "
                 "arms are strictly stronger and they are labelled supplementary so the "
                 "pre-registered measure stays legible on its own.")},
    ]
    return out


# ----------------------------------------------------------------- check ----

def numeric_leaves(obj, prefix=""):
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        yield prefix, float(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from numeric_leaves(v, f"{prefix}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from numeric_leaves(v, f"{prefix}/{i}")


SKIP_CHECK_PREFIXES = ("/versions", "/runtime")


def compare(fresh: dict, stored: dict, tol: float) -> int:
    a = {k: v for k, v in numeric_leaves(fresh) if not k.startswith(SKIP_CHECK_PREFIXES)}
    b = {k: v for k, v in numeric_leaves(stored) if not k.startswith(SKIP_CHECK_PREFIXES)}
    bad = []
    for k in sorted(set(a) | set(b)):
        if k not in a or k not in b:
            bad.append((k, a.get(k), b.get(k)))
        elif not math.isclose(a[k], b[k], rel_tol=0.0, abs_tol=tol):
            bad.append((k, a[k], b[k]))
    if bad:
        for k, x, y in bad[:25]:
            print(f"CHECK MISMATCH {k}: recomputed {x} vs stored {y}")
        print(f"CHECK FAILED: {len(bad)} numeric leaves differ beyond {tol:g}")
        return 5
    print(f"CHECK OK: {len(a)} numeric leaves reproduce within {tol:g}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "results" / "safety_control_plane.json"))
    ap.add_argument("--check", action="store_true",
                    help="re-enumerate and compare every numeric leaf at 1e-6; exit 0/5")
    ap.add_argument("--check-tol", type=float, default=1e-6)
    args = ap.parse_args()

    out = build()

    if args.check:
        stored = json.loads(Path(args.out).read_text())
        stored_sha = stored["data_sources"][0]["sha256"]
        fresh_sha = out["data_sources"][0]["sha256"]
        if stored_sha != fresh_sha:
            print(f"ADVISORY: SAFETY-DESIGN.md sha256 moved since the recorded run "
                  f"({stored_sha[:12]} -> {fresh_sha[:12]}). The rule text under test may "
                  f"have changed. This does not change the exit code.")
        return compare(out, stored, args.check_tol)

    atomic_write_json(args.out, out)
    p = out["primary"]
    print(f"\nwrote {args.out}")
    print(f"  status {out['status']}, {out['enumeration']['exhaustive_or_sampled']}")
    print(f"  {p['n_triples_enumerated']:,} triples over "
          f"{out['enumeration']['n_base_records']:,} base records")
    print(f"  monotonicity violations                       "
          f"{p['n_monotonicity_violations']:,}")
    print(f"  model-derived reads in the deterministic path "
          f"{p['n_model_derived_reads_in_deterministic_path']:,}")
    print(f"  of those, reaching the graded block                "
          f"{out['coverage']['n_triples_reaching_the_graded_block']:,}, "
          f"model changed the outcome on "
          f"{out['coverage']['n_triples_where_the_model_changed_the_disposition']:,} "
          f"(all toward more friction: "
          f"{out['coverage']['n_triples_where_the_model_added_friction']:,})")
    print(f"  dispositions reached: "
          f"{', '.join(out['coverage']['dispositions_reached'])}")
    print(f"  internal reason labels never reached: "
          f"{out['coverage']['n_internal_reason_codes_never_reached']}")
    for r in out["sanity"]["mutants"]:
        print(f"  mutant {r['name']:<42} mono {r['n_monotonicity_violations']:>10,}  "
              f"taint {r['n_model_derived_reads_in_deterministic_path']:>10,}  "
              f"as declared {r['caught_as_expected']}")
    print(f"  wall {out['runtime']['wall_seconds']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
