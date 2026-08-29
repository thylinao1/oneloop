# Pre-registration F-B: extending the fraud headroom curve (finals menu, last item)

Status: written on 2026-08-25 under the approved finals menu (2026-08-24). It became a
pre-registration when committed to main unchanged. Runs only on the shortlist
trigger and only if queue time remains after F-D, F-C and F-A. Results ship as obtained, in
finals material only; the uploaded Round 1 artifact is never modified.

## Question

The shipped scaling exhibit shows the fraud PR-AUC gain from embeddings large where the baseline
has headroom (3M rows: 0.485 to 0.551; 10M rows: 0.638 to 0.904, interval [0.2001, 0.3436]) and a
null at the full corpus where the baseline is saturated. Where does the gain fall off? Two added
corpus-size points trace the curve.

## Design, fixed now

- Corpus-size points: 1,000,000 and 20,000,000 prepared rows, earliest-window subsets built the
  way the shipped 3M and 10M points were built (each point recomputes its own time cut and
  vocabulary from the rows it holds, the same disclosed limitation the shipped card carries).
- Recipe and protocol frozen: the shipped pretrain configuration per point, the shipped L3
  transfer evaluation, the same row caps (800,000 train, 300,000 test) where they bind, entity
  clustered paired bootstrap B = 1000.
- No new model, no changed hyperparameter, no changed evaluation.

## Decision rule, frozen verbatim

There is no win condition. The curve ships as obtained, added to the finals material as two more
points on the existing exhibit, with the fraud positive count per point printed beside each.
If the new points break the headroom story (for instance the 1M point shows no gain, or the 20M
point shows one on a saturated baseline), the finals sentence says so in those words.

## Corpus and provenance

Synthetic IBM TabFormer, labelled synthetic. The points are nested subsets of one corpus and are
not independent samples; the sentence says so.

## Output and cost

results/scale/scale-1m.json and results/scale/scale-20m.json merged under tags `scale-1m` and
`scale-20m` by scripts/merge_scaling.py. Two pretrains at reduced corpus size plus two evaluation
passes; the 20M pretrain is close to the shipped main run in cost. Exact accounting from Slurm
after the run.
