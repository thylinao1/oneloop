# Frozen results

Every figure in the entry is bound to a path in one of these files, and the interactive entry opens the file at the line the number sits on. A result, once produced under its pre-registration, is not regenerated to read better; the wrong-side results are still here.

| File | Produced by | Carries |
|---|---|---|
| `ladder.json` | `scripts/fm/ladder_eval.py` | the four-rung leakage ladder |
| `backbone_transfer.json` | `scripts/fm/transfer_eval.py` | multi-task transfer with paired intervals |
| `cj/*.json` | `scripts/fm/*_cj.py` | the real-corpus replication, two seeds |
| `protection.json`, `protection_queue.json` | `scripts/fm/protection_pll.py` | label-free fraud scoring against the counting control |
| `uplift.json`, `uplift_shares.json`, `uplift_wasted_budget.json`, `uplift_subgroup.json` | `scripts/uplift_*.py` | the offers head on randomized Criteo and Hillstrom |
| `corridor.json`, `corridor_combination.json`, `corridor_totals.json` | `scripts/corridor_*.py` | the corridor head on SingStat arrivals |
| `whitespace.json`, `whitespace_temporal.json`, `whitespace_stratified.json`, `whitespace_control.json` | `scripts/whitespace_*.py` | the signing head and its forward check |
| `narratives.json`, `narratives_direction.json`, `narratives_support.json` | `scripts/narratives/*` | the generated-text layer and its audits |
| `safety_*.json` | `scripts/safety/*` | privacy ladder, membership inference, injection red team, inversion, control plane |
| `scale/*.json` | `scripts/scale/*` | the scaling curve and the merchant axis |
| `pair_retention.json` | `scripts/pair_retention.py` | the merchant-native task (a null) |
| `atlas.json` | `scripts/scale/atlas_build.py` | the merchant embedding atlas sample |
| `cost_model.json` | `scripts/cost_model.py` | the metered compute bill |
| `protocol_value.json` | `scripts/protocol_value.py` | the 27-comparison scoreboard |
| `value_model_sensitivity.json` | `scripts/value_model_sensitivity.py` | one-way sensitivity on the value model |
| `whitespace_map_points.json` | `scripts/whitespace_exhibit.py` | the map sample behind the signing exhibit |
| `reading_path.json` | | page boundaries of the submitted document |
