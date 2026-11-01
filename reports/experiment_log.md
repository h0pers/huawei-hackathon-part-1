# Experiment log

Validation is each user's first event in Jan 24-26, with everything fitted before Jan 24 ("v24").
"v26" is the same on Jan 26, with six days of data. Unless a line says otherwise, numbers are v24
and the softmax model alone.

## Submissions

| File | v24 | Public |
|---|---|---|
| Old pipeline | 0.80790 | 0.824902 |
| Softmax alone | 0.80813 | 0.822332 |
| Softmax + LightGBM | 0.81005 | 0.824422 |
| 60/40 rank blend of the two stacks | 0.81171 | 0.825683 |
| **Joint LightGBM (`submission_joint.csv`)** | **0.81305** | **0.827606** |

v24 misranks the old pipeline against the new models: it is last locally and second on public.
The other four keep their order, and their gaps to public are steady (0.0140 to 0.0146).

## Kept

| Change | Result |
|---|---|
| Softmax MF with an outside option, first fit | 0.8056 |
| Warm-up clicks as a partial likelihood | Essential: 0.68 without them |
| ALS item vectors as the start | Essential: 0.62 from random |
| Strong item bias prior (`l2_bias` 0.1 to 10) | +0.0016 |
| Norm caps (u about 2, v about 2.7) | +0.0012; the best cap barely moves on v26 |
| More epochs at a lower step size | +0.0003 |
| Old and softmax features in one LightGBM | Best: v24 0.81305, v26 0.83467 |

## Lost

| Bet | Result | Baseline |
|---|---|---|
| Variational Bayes | 0.796 | 0.8058 |
| Bootstrap ensemble, 30 members | 0.8046 | 0.8058 |
| Same-data seed ensemble, 4 members | 0.8065 | 0.8058 (small gain, unused) |
| History pooled into the user vector | 0.799 | 0.8058 |
| Taste dimension 6, 7, 9, 12, 16, 24 | 0.77 to 0.80 | 0.8069 at 8 |
| Repeat-click terms | No effect | |
| Per-user sphere posterior | 0.8066 | 0.8088 |
| Alternating posterior users and item refits, replica | 0.825 | 0.829 |
| Annealing the prior or caps, 13 variants | 0.8075 to 0.8088 | 0.8088 |
| Softmax item consensus with refitting | 0.8083 to 0.8087 | 0.8086 |
| Cross-fitting items and users | 0.8055 | 0.8088 |
| Frequency-scaled item prior | 0.8086 to 0.8088 | 0.8088 |
| Graph smoothing of items, best | 0.8090 | 0.8088 (within noise) |
| Refit users on frozen items | 0.8088 | 0.8088 |
| ALS factors 10, 16, 20; alpha 100 | 0.8084, 0.7866, 0.7620, 0.8083 | 0.8088 |
| Temperature 1.4 | +0.0003 | Superseded by LightGBM |
| Log-impressions term, v26 | 0.8297 to 0.8299 at best | 0.8300 |
| Late-event weighting, all-event LightGBM training | 0.8102 to 0.8106 | 0.8105 stack |
| LightGBM settings on the joint features, best | 0.81317 | 0.81300 joint |
| Joint + ALS-seed, cosine, posterior, dim-10 features | 0.81353 (+0.0005) | 0.81300 joint |

## Data findings

- Fitted user and item vectors look uniform on the unit sphere in 8 dimensions.
- Candidates are in the user's click history 2.5% of the time, against 0.04% by chance, and those rows have a 0.298 click rate against 0.057. Otherwise candidates carry no taste signal.
- Re-shown non-clicked items have no extra signal (0.0588 click rate).
- All 761,468 in-period history clicks match labelled train clicks. None falls after the cutoff.
- User click propensity is stable: 0.61 correlation between the two halves of the week.
- Older warm-up clicks stay useful: AUC falls steadily as they are dropped (0.8088 to 0.7527 at three days).
- The noise in a public score is about 0.002 alone and 0.0006-0.0008 between two models, assuming a 20-30% public share. The real share is unknown.

## Replica of the simulator

Synthetic data on the real event structure with known sphere vectors, matched at concentration
about 6.5. It rests on the inferred sphere model and is only a screening tool.

| Setup at 6.6 | AUC |
|---|---|
| True users and items | 0.914 |
| True users, learned items | 0.883 |
| True items, learned users | 0.864 |
| Start from truth, then train both | 0.848 (0.910 after one epoch) |
| Start from true items, then train | 0.835 |
| Our model | 0.829 |

- Training away from the truth lowers AUC, so the objective itself overfits.
- With true items, a Bayesian user posterior (0.8626) does no better than the point estimate (0.8643), so about 0.864 is the realistic upper bound here. It requires perfect items.
- Item recovery on the replica runs from 0.66 under 5 clicks to 0.97 at 40-80, and each +0.01 similarity to truth is worth about +0.0035 AUC.
- Each item bet moved the replica and real validation the same way.
- Real and replica windows differ by under 0.002 on v24-v26. Public is about 0.003 below the replica, which is roughly one standard deviation of public noise once older settings are allowed for, so there is no solid evidence of a shift at the cutoff.

## Reproduction

The repo pipeline replaced the scratch scripts that produced the uploaded file.

- Validation: 0.81299, and 0.81298 after removing unused options, against 0.81305 for the scratch version.
- Its evaluation ranking correlates 0.99975 with the uploaded `submission_joint.csv`. The files are not byte-identical, and the rerun has not been scored on public.

## Candidate not recommended

`submission_joint_plus.csv` adds the ALS-seed, cosine and posterior features: v26 0.83501 against 0.83466, and about +0.0005 on v24. That is below the 0.0006-0.0008 public noise between two models, so it does not meet the bar for a submission. It came from scratch scripts and is not in this pipeline.
