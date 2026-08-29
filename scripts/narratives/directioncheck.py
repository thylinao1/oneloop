"""Deterministic DIRECTION layer for the GenAI narratives exhibit.

Why this exists
---------------
The numeric layer (faithcheck.py) proves every numeral in a narrative matches a fact value.
It cannot prove the narrative points those numerals the right way, and that gap shipped:
five of the ten corridor narratives said the model "outperformed" the seasonal naive while
their own facts said the naive was ahead, and all five carried both a numeric pass and an
LLM cross-examination pass. Every numeral in them was correct. The comparison was reversed.

The cause was a fact bundle, not a hallucination. make_inputs.py told the generator
"MASE below 1 beats the seasonal-naive baseline", and every reversed narrative had a model
MASE below 1, so the model applied the rule it was handed. The LLM cross-examination could
not catch it either, because it was given the same wrong premise and is the same model.

That is the general lesson worth more than the bug: a gate that checks whether numbers are
RIGHT is not a gate that checks whether the sentence about them is right. This layer closes
that, deterministically, with no model in it.

How it works
------------
For each comparison this project actually makes, a rule declares which two fact fields are
being compared and which direction is better. The truth is computed from the numbers. The
narrative is then scanned for comparative language, and if it makes a directional claim that
disagrees with the numbers, that is a failure naming both values.

Deliberately narrow. It fires only on comparisons declared in COMPARISONS and only on the
verbs in _WIN / _LOSE. It is not a general claim checker and does not pretend to be one; a
narrative making a comparison this table does not know about is reported as unchecked rather
than silently passed, so the coverage is visible instead of assumed.
"""

import re

# (label, better_field, worse_field, polarity)
#   polarity "lower" -> the smaller value is the better one (error metrics)
#   polarity "higher" -> the larger value is the better one (AUC, recall, lift)
# "subject" is the thing the narrative calls ours; "baseline" is what it is compared against.
COMPARISONS = [
    {
        "label": "corridor MASE against the seasonal naive",
        "subject": "mase_model",
        "baseline": "mase_seasonal_naive",
        "polarity": "lower",
    },
]

_WIN = re.compile(
    r"\b(outperform\w*|out-perform\w*|beat\w*|better than|superior to|improv\w+ (?:on|over)|"
    r"exceed\w* the baseline|ahead of the (?:baseline|naive))\b", re.I)
_LOSE = re.compile(
    r"\b(underperform\w*|under-perform\w*|lost to|loses to|worse than|behind the (?:baseline|naive)|"
    r"fail\w* to beat|did not beat|does not beat)\b", re.I)


def _truth(facts, rule):
    a, b = facts.get(rule["subject"]), facts.get(rule["baseline"])
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return None
    if isinstance(a, bool) or isinstance(b, bool):
        return None
    return (a < b) if rule["polarity"] == "lower" else (a > b)


def direction_failures(narrative, facts):
    """Return a list of direction failures. Empty means nothing contradicted."""
    out = []
    text = narrative or ""
    for rule in COMPARISONS:
        subject_wins = _truth(facts, rule)
        if subject_wins is None:
            continue
        says_win = bool(_WIN.search(text))
        says_lose = bool(_LOSE.search(text))
        if says_win == says_lose:
            # Either no directional claim at all, or both directions appear and the sentence
            # is doing something this layer will not guess at. Neither is a contradiction.
            continue
        claimed_win = says_win
        if claimed_win != subject_wins:
            a, b = facts.get(rule["subject"]), facts.get(rule["baseline"])
            better = "lower" if rule["polarity"] == "lower" else "higher"
            out.append(
                f"{rule['label']}: the narrative says the subject "
                f"{'beat' if claimed_win else 'lost to'} the baseline, but "
                f"{rule['subject']}={a} against {rule['baseline']}={b} and {better} is better, "
                f"so the subject actually {'beat' if subject_wins else 'lost to'} it"
            )
    return out


def is_checkable(facts):
    """True when at least one declared comparison applies to this bundle.

    Reported alongside the pass count so coverage is visible: a bundle nothing here understands
    is unchecked, not passed, and saying so is the difference between a gate and a decoration.
    """
    return any(_truth(facts, rule) is not None for rule in COMPARISONS)
