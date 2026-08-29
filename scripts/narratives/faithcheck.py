"""Deterministic numeric faithfulness layer for the GenAI narratives exhibit.

Shared by generate.py (cluster, at generation time) and check.py (Mac, --check).
Pure Python stdlib, CPU, deterministic. Base tolerance 1e-6.

Rule: every numeral extracted from a narrative must match some numeric fact
value from its input_facts bundle -- exact within 1e-6, rounding-tolerant
(the narrative may round a fact to the precision it states), and
percent-aware (fact 0.045 may appear as 4.5 when clearly a percentage,
and vice versa).

Where the allowed pool comes from, and why it changed. The first version built
the pool by scanning every fact string, so whoever controlled one string field
could write digits into the pool and have a false figure accepted. Our own red
team measured that path at 30 of 30 bundles (results/safety_injection.json,
section c2_pool_pollution). The pool is now taken from numeric-typed leaves of
the parsed bundle. Free text is not scanned, except for the small set of fields
listed in _TRUSTED_STRING_FIELDS, which narratives quote and which are either
string constants written in our own source or values pinned to a declared
format. The pre-fix builder stays in this file as
collect_fact_numbers_string_scan so the red-team scripts can replay the
published measurement against the code it was taken on.
"""

import re

BASE_TOL = 1e-6

# numerals incl. thousands separators and decimals: 3,100,145 / 0.904 / 13 / -0.61
_NUM_RE = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+\.\d+|-?\d+")

# The only fields whose STRING value may add numerals to the allowed pool.
#
# None means the value is a string constant written in our own source and is
# never assembled from data, so its numerals are fixed when we write them:
#   what               scripts/narratives/make_inputs.py:38, :64, :99
#   attribution_label  scripts/corridor_exhibit.py:577
#   verdict_rule       scripts/uplift_exhibit.py:448
#
# A pattern means the value carries data, so the WHOLE value must match the
# format our pipeline writes before any numeral in it counts. Text appended to
# the field, which is the injection shape the red team used, breaks the match
# and the field then adds nothing at all. The name slots inside the whitespace
# reason templates reject digits, so a label written into `area` or `category`
# cannot pass a figure through a reason string either.
#
# Adding a field here widens what the gate will accept. Add one only when it is
# a source constant or when its format pins every numeral it can carry.
_TRUSTED_STRING_FIELDS = {
    "what": None,
    "attribution_label": None,
    "verdict_rule": None,
    # model feature column name: lag1, lag12, roll12, corridor_id
    "feature": re.compile(r"[a-z][a-z0-9_]*"),
    # Hillstrom segment name: 3) $200 - $350 x recency 1-3m
    "segment": re.compile(r"\d+\) \$[\d,]+ - \$[\d,]+ x recency \d+-\d+m"),
    # whitespace reason lines, written by scripts/whitespace_exhibit.py:431-446
    "reasons": re.compile(
        r"MDR-sensitive category mix \([^\d()\[\]]+, prior \d+\.\d+: "
        r"\d+(?:\.\d+)?-\d+(?:\.\d+)?% card MDR vs \d+(?:\.\d+)?% QRIS anchor\)"
        r"|Premium-demand corridor \(\d+\.\d+ km to [^\d()\[\]]+; "
        r"zone score \d+\.\d+\)"
        r"|Dense commercial cluster \(median \d+ POIs within \d+ m\)"
        r"|\d+% independent \(non-chain\) merchants"),
}


def extract_numerals(text):
    """Return list of (float_value, decimals_stated) for every numeral in text."""
    out = []
    for m in _NUM_RE.finditer(text):
        raw = m.group(0).replace(",", "")
        val = float(raw)
        dec = len(raw.split(".")[1]) if "." in raw else 0
        out.append((val, dec, m.group(0)))
    return out


def trusted_string_numbers(field, text):
    """Numerals a string contributes, which is nothing unless its field is trusted.

    `field` is the dict key the string sits under. A trusted field that carries
    data only contributes when the whole value matches its declared format.
    """
    if field not in _TRUSTED_STRING_FIELDS:
        return []
    pattern = _TRUSTED_STRING_FIELDS[field]
    if pattern is not None and pattern.fullmatch(text) is None:
        return []
    return [v for v, _d, _r in extract_numerals(text)]


def collect_fact_numbers(obj, field=None):
    """Build the allowed numeral pool for a facts bundle.

    Numbers come from numeric-typed leaves of the parsed bundle. A string adds
    numerals only through trusted_string_numbers, so text written into a data
    field cannot widen the pool the gate checks against.

    `field` is the dict key the value sits under; list items keep the key of the
    list that holds them.
    """
    vals = []
    if isinstance(obj, bool):
        return vals
    if isinstance(obj, (int, float)):
        vals.append(float(obj))
    elif isinstance(obj, str):
        vals.extend(trusted_string_numbers(field, obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            vals.extend(collect_fact_numbers(v, k))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            vals.extend(collect_fact_numbers(v, field))
    return vals


def collect_fact_numbers_string_scan(obj):
    """The pre-fix pool builder: every numeral in every fact string counts.

    Kept so the red-team scripts replay the published measurement in
    results/safety_injection.json against the code that measurement was taken
    on. Do not use it in the gate. One string written into one data field
    enlarges the pool, which is the hole the red team hit on 30 of 30 bundles.
    """
    vals = []
    if isinstance(obj, bool):
        return vals
    if isinstance(obj, (int, float)):
        vals.append(float(obj))
    elif isinstance(obj, str):
        vals.extend(v for v, _d, _r in extract_numerals(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            vals.extend(collect_fact_numbers_string_scan(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            vals.extend(collect_fact_numbers_string_scan(v))
    return vals


def _matches(n, dec, fact):
    """Does narrative numeral n (stated with `dec` decimals) match fact value?"""
    for c in (fact, fact * 100.0, fact / 100.0):  # percent-aware
        if abs(n - c) <= BASE_TOL:
            return True
        # rounding-tolerant: the narrative rounded the fact to `dec` decimals
        if abs(n - round(c, dec)) <= BASE_TOL:
            return True
    return False


def support_rank(n, dec, fact) -> int:
    """How closely a fact supports a printed numeral. Lower is closer.

    _matches answers only "is this numeral carried by some fact", which is the gate's
    question. Anything that NAMES the supporting fact needs to choose between several
    facts that all match, and _matches is percent-aware, so an unrelated fact can match
    through a factor of a hundred. Ranking the candidates lets a caller name the closest
    one. This adds no call path inside the gate and changes no published verdict.

      0  the fact exactly as stored
      1  the fact rounded to the precision the narrative printed
      2  the same fact read as a percentage
      3  read as a percentage and rounded as well
    """
    if abs(n - fact) <= BASE_TOL:
        return 0
    if abs(n - round(fact, dec)) <= BASE_TOL:
        return 1
    for scaled in (fact * 100.0, fact / 100.0):
        if abs(n - scaled) <= BASE_TOL:
            return 2
    return 3


def unsupported_numerals(narrative, facts, pool_fn=collect_fact_numbers):
    """Return list of numeral strings in `narrative` with no supporting fact.

    `pool_fn` exists for the red-team replay scripts, which pass
    collect_fact_numbers_string_scan to reproduce the published measurement.
    The gate always takes the default.
    """
    fact_vals = pool_fn(facts)
    bad = []
    for n, dec, raw in extract_numerals(narrative):
        if not any(_matches(n, dec, f) for f in fact_vals):
            bad.append(raw)
    return bad
