# Approach

Team-maintained. Update it when the direction changes.

## Organiser baseline
`starter_solution.py` (standard library only, not in this repo):
- Smoothed item CTR and user CTR (pseudo-count 50 toward the global rate), combined in logit space.
- An item-item co-click score between the candidate and the user's history (cosine-normalised),
  added with weight 25.
- The organisers say it is "clearly beatable". They suggest that modelling user×item interactions
  directly, e.g. with matrix factorization, should do meaningfully better.

## Current direction
- Data pipeline to Parquet first (in progress, see `docs/data.md`).
- Then a validated reimplementation of the baseline as the reference score.

## Ideas not tried yet
- User/item embeddings (MF or two-tower) trained on train clicks and histories.
- Gradient-boosted trees on CTR, co-click, recency and history-similarity features.
- Within-event features: candidate rank or score relative to the other 9 candidates.
- Cold-start fallbacks for unseen users: item popularity plus history-based signals.