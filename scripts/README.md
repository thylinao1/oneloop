# Pipeline map

Each producer writes one file into `results/` and embeds its seed, library versions and data provenance in the file it writes. The stages below are listed in pipeline order.

## 1. Data

| Script | Writes | Note |
|---|---|---|
| `fetch_data.sh` | `data/` | fetches the public corpora with checksums |
| `fsq_slice.py` | SG slice of Foursquare OS Places | streams the 239GB dataset with predicate pushdown; the full download is never taken |
| `fm/prep.py` | tokenized TabFormer prep | vocabulary built with the label column excluded |
| `fm/prep_cj.py` | tokenized Complete Journey prep | same recipe on the real retail corpus |

## 2. Backbone

| Script | Writes | Note |
|---|---|---|
| `fm/model.py`, `fm/pretrain.py` | checkpoint | field-masked transformer, 20.5M parameters, 8 epochs |
| `fm/embed.py` | entity embeddings | as-of, prefix-only construction |
| `fm/job_backbone.sbatch` and kin | | Slurm entry points per stage |

## 3. Evaluation

| Script | Writes | Note |
|---|---|---|
| `fm/ladder_eval.py` | `results/ladder.json` | the four-rung leakage ladder; one frozen checkpoint under four protocols |
| `fm/transfer_eval.py` | `results/backbone_transfer.json` | multi-task transfer with entity-level paired intervals |
| `fm/transfer_eval_cj.py`, `fm/cj_*.py` | `results/cj/*` | the pre-registered replication on a real corpus, two seeds |
| `fm/protection_pll.py` | `results/protection.json` | label-free surprise score against a counting control |
| `protection_queue.py` | `results/protection_queue.json` | the same result restated in review-queue units |
| `scale/*` | `results/scale/*` | the scaling curve and the merchant-axis run |
| `pair_retention.py` | `results/pair_retention.json` | the merchant-native task; the result is a null and ships as one |

## 4. Heads

| Script | Writes | Note |
|---|---|---|
| `uplift_exhibit.py` | `results/uplift.json` | offers head on randomized Criteo; Qini, targeting at k, paired differences |
| `uplift_shares.py`, `uplift_wasted_budget.py`, `uplift_subgroup.py` | `results/uplift_*.json` | the headline in budget-owner units, the wasted-budget measurement, the subgroup audit |
| `corridor_exhibit.py` | `results/corridor.json` | corridor head on SingStat arrivals; MASE against a per-corridor seasonal naive |
| `corridor_combination.py` | `results/corridor_combination.json` | the pre-registered naive-plus-model blend at fixed equal weights |
| `corridor_totals.py` | `results/corridor_totals.json` | the same error totalled in arrivals, the view that reverses |
| `whitespace_exhibit.py` | `results/whitespace.json` | signing head; real-signals ranking of pseudonymized SG merchant buckets |
| `whitespace_temporal.py`, `whitespace_stratified.py`, `safety/whitespace_control.py` | `results/whitespace_*.json` | the forward check and its controls |

## 5. Generated-text layer

| Script | Writes | Note |
|---|---|---|
| `narratives/make_inputs.py`, `narratives/generate.py` | `results/narratives.json` | narratives written by a self-hosted model over committed fact bundles |
| `narratives/faithcheck.py` | | layer one: every numeral in the text must be carried by a fact |
| `narratives/check.py` | | layer two: model cross-examination of each claim |
| `narratives/directioncheck.py` | `results/narratives_direction.json` | layer three: the direction of each comparison |
| `narratives/support_audit.py` | `results/narratives_support.json` | how much weight the permissive part of layer one carries |

## 6. Safety

| Script | Writes | Note |
|---|---|---|
| `safety/privacy_ladder.py` | `results/safety_privacy_ladder.json` | each guard priced one rung at a time |
| `safety/mia_eval.py` | `results/safety_membership.json` | membership inference at the account level, with the no-model control |
| `safety/injection_corpus.py`, `safety/injection_run.py`, `safety/injection_redteam.py` | `results/safety_injection.json` | seven attack classes against the narrative gate, corpus frozen before scoring |
| `safety/test_injection_fix.py` | `results/safety_injection_fix.json` | the pool-pollution hole, fixed and replayed |
| `safety/embedding_inversion.py` | `results/safety_embedding_inversion.json` | what an attacker recovers from a leaked embedding |
| `safety/control_plane_conformance.py` | `results/safety_control_plane.json` | conformance checks on the designed control plane |

## 7. Accounting

| Script | Writes | Note |
|---|---|---|
| `cost_model.py` | `results/cost_model.json` | the compute bill from cluster accounting, failed runs included |
| `protocol_value.py` | `results/protocol_value.json` | the 27-comparison scoreboard, drawn from the pre-registered record |
| `value_model_sensitivity.py` | `results/value_model_sensitivity.json` | one-way sensitivity on the bottom-up value model |
