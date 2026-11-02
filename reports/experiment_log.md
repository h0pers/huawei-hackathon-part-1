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

## Research loop from 2026-11-01

Scripts are in `research/`. `hypotheses.md` holds each hypothesis. Baseline rerun: v24 0.81298, v26 0.83451.

### Validation check against public

v24 orders four of the five past submissions correctly. It misplaces the old pipeline (last locally, second on public), as already recorded above. Not yet fixed. Treat any gain that favours older features over softmax with suspicion.

### Round 1: forensics, no modelling (`f01`, `f02`, `f03`)

| Probe | Result | Verdict |
|---|---|---|
| Raw file position of the clicked row | Uniform, p=0.84 | No position signal |
| Catalogue change | 99% of items appear within 0.57 days, 99% are still shown after day 6.76 | Static, no new items |
| Drift: similarity AUC by lag | 0.683 at 1-2 days, 0.673 at 15-25 days | Essentially none |
| Context: same vs other bucket or day type | 0.677 vs 0.679; 0.679 vs 0.677 | None |
| Global context taste by bucket | About the null level | None |
| Multi-interest: kernel-weighted history | Mean 0.7755, every kernel lower, max 0.713 | Single interest |
| Repeat click rate by time since click | Flat about 0.29 from 1h to 14d+ | No Hawkes or fatigue decay |
| Re-show probability by history slot | Geometric, newest slot 0.217 | Candidate generator favours recent clicks; the click rate does not |
| Exposure | Heavy tail 25-1739, early/late corr 0.93, no link to direction (R2 0.006) | Exposure is an item property |
| New-row click rate by exposure | 0.049 to 0.062 | Small |
| ICA on users and items | All kurtosis about -0.7 | Isotropic, no hidden axes or clusters |
| Propensity if known exactly | Event AUC 0.671 by moments; stack has 0.669 | Not a separate lever |

### Round 1 diagnostics on the baseline (v24)

| Component | Value |
|---|---|
| Global | 0.81298 |
| Within-event | 0.80115 |
| Between-event (sum of p) | 0.66898 |
| Global with perfect event knowledge | 0.88897 |
| Global with perfect within-event ranking | 0.96366 |

| Segment | AUC |
|---|---|
| Unseen / seen users | 0.738 / 0.817 |
| History empty, 1-4, 5-9, 10-19, 20 | 0.561, 0.738, 0.790, 0.821, 0.840 |
| Candidate new / in history | 0.802 / 0.680 |
| Item impressions <40 to >200 | 0.768, 0.790, 0.810, 0.824, 0.838 |
| Since last click <6h to >7d | 0.841, 0.833, 0.814, 0.793, 0.750 |
| User events 0 to 20+ | 0.738, 0.778, 0.809, 0.829, 0.842, 0.852 |

AUC rises steadily with data per user and per item, which points to an estimation limit. Evaluation users have more events than v24 users (19% vs 8% have 20+), which may partly explain why public beats v24.

### Round 2: spectral item vectors (H1). Killed.

Exposure-corrected SVD and correspondence analysis, against ALS history similarity of 0.7755: best 0.574. The spectrum is flat. Schnabel et al. 2016 and webinar slide 6 motivated the test.

### Round 3: joint posterior averaging (H2)

Softmax alone, v24, MAP = 0.8088. "Limit" fits A + c/k to the running average and is an extrapolation, not a measurement. Chains are not fully comparable: lengths differ.

| Chain | Samples | Average | Limit |
|---|---|---|---|
| Plain SGLD, T=1, eps 1e-3 | 53 | 0.7619 | Mixing too slow |
| Preconditioned, ball, T=1 | 220 | 0.80448 | 0.8086 |
| Preconditioned, ball, T=0.3 | 220 | 0.80994 | 0.8114 |
| Preconditioned, ball, T=0.3, 4000 steps | 360 | 0.81037 | |
| Sphere k=6.5, T=1 | 260 | 0.80670 | 0.8097 |
| Sphere k=6.5, T=0.3 | 260 | 0.80761 | |
| Sphere k=4.5, T=1 | 260 | 0.80199 | |
| Sphere k=9, T=1 | 260 | 0.80702 | |

### Round 3 in the stack

Chain features added to the joint LightGBM: averaged probability, averaged score, score standard deviation across samples, score vs event mean, rank, no-click probability.

| Model | Global | Within-event | Between-event |
|---|---|---|---|
| Joint baseline, v24 | 0.81298 | 0.80115 | 0.66898 |
| Joint + chain (T=0.3, 4000 steps), v24 | 0.81414 | 0.80231 | 0.67018 |

Early stopping used the v24 window itself (467 rounds), so this gain has a small optimistic bias. v26 confirmation is running.

v26 confirmation: softmax alone 0.8300 to 0.8314; joint 0.83451 to 0.83539 (+0.00088). Direction agrees with v24, though each gain is near the 0.0006-0.0008 noise between two models on public.

Tuning on softmax alone, v24, 4000 steps: T=0.15 0.81034, T=0.3 0.81037, T=0.5 0.80976, T=0.3 with eps 0.2 0.81057. Flat, so temperature is not the lever.

### Candidate: `models/submission_chain.csv`

Joint + chain, T=0.3, 4000 steps, 520 rounds. Valid against raw evaluation.csv (`research/validate_submission.py`). Spearman 0.99551 with the 0.827606 file. Expected public about 0.8285 (range 0.828-0.829). Public score: **0.828145** (+0.00054 over 0.827606). Low end of the predicted range, and within the 0.0006-0.0008 noise between two models.

### Round 4: what LightGBM adds above the chain (`r11_ablate.py`, v24)

Under a correct model the posterior average should be close to optimal, yet LightGBM adds +0.0037. Chain-only LightGBM 0.81066, all features 0.81408.

| Family | Added to chain | Removed from all |
|---|---|---|
| ALS | 0.81267 (+0.0020) | 0.81215 (-0.0019) |
| History similarity | 0.81117 | 0.81398 |
| MAP softmax | 0.81115 | 0.81389 |
| Covis | 0.81103 | 0.81392 |
| User activity | 0.81091 | 0.81377 |
| Norms | 0.81092 | 0.81398 |
| Repeat | 0.81086 | 0.81394 |
| Old choice | 0.81086 | 0.81402 |
| User rate | 0.81080 | 0.81409 |
| Item stats | 0.81074 | 0.81387 |
| Context | 0.81063 | 0.81392 |

ALS is the only family with a large unique contribution, and it acts within events (0.7992 to 0.8016).

Multi-start chains so far: starts 42 and 1000 give 0.80992 and 0.81005, average 0.81056. That is about what doubling the samples predicts, so no evidence of extra modes yet.

### Round 5: chains from several starts (H3). Killed.

Softmax alone, v24, 3000 steps per chain: starts 42, 1000, 2000, 3000 give 0.80992, 0.81005, 0.81009, 0.81010; all four averaged 0.81092. A single 12000-step chain gives 0.81091. So the extra starts add only samples. The posterior limit from this start family is about 0.811, below the earlier 0.8114 extrapolation.

### Round 6: likelihood weights. Killed, and informative.

Softmax MAP, v24, weights on no-click events, clicked events and warm-up clicks.

| wn, wc, ww | AUC |
|---|---|
| 1, 1, 1 | 0.80880 |
| 0.5, 1, 1 | 0.80802 |
| 0.2, 1, 1 | 0.80530 |
| 0, 1, 1 | 0.80029 |
| 1, 1, 2 | 0.80521 |
| 1, 1, 0.5 | 0.80454 |
| 1, 0.5, 1 | 0.80577 |
| 1, 2, 1 | 0.80445 |
| 0.5, 1, 2 | 0.80406 |
| 0.5, 2, 2 | 0.80542 |

Equal weights are best, which supports the softmax likelihood with an outside option and warm-up partial likelihood.

### Round 7: which ALS column (`r13`, v24)

Chain-only LightGBM 0.81066. Added singly: `als_user_score` 0.81245, `als_hist_rank` 0.81083, `als_hist_sim` 0.81077, `als_last_sim` 0.81069. ALS alone by seed count (`r12`): cosine 0.77065 at 1 seed, 0.77721 at 5, 0.77873 at 10, 0.77888 at 20; dot 0.73711, 0.74163, 0.74268, 0.74279. Item norm alone 0.533.

### Round 8: ALS cosine by factor count (`r15`, `r16`, v24)

Chain-only LightGBM (7 columns with the new one) against all-features 0.81408, fast settings.

| Spec | Alone cos | Chain + cos | Chain + dot |
|---|---|---|---|
| 8 factors | 0.77926 | 0.81193 | 0.81122 |
| 12 factors | 0.77873 | 0.81396 | 0.81262 |
| 16 factors | 0.75215 | 0.81426 | 0.81324 |
| 24 factors | 0.73968 | 0.81422 | 0.81344 |
| 12 factors, alpha 100 | 0.77032 | 0.81289 | 0.81197 |
| 12 factors, alpha 1500 | | 0.81361 | |
| 12 factors, reg 3 | 0.77875 | 0.81393 | 0.81257 |

Higher factor counts score worse alone but add more, so the extra factors hold something the rank-8 softmax does not represent. Not yet identified.

Full stack, standard settings: joint + chain 0.81414; + f16 cos 0.81509; + f12, f16, f24 cos 0.81524 (+0.0011). These ALS fits use the fit labels and target histories only, as `click_log` does. v26 confirmation pending.

v26 confirmation of the ALS cosine columns: joint + chain 0.83539 to 0.83633 (+0.00094), 508 rounds. v24 gave +0.0011 at 420 rounds. Submission uses 460 rounds.

### Round 9: what the extra factors capture (H5, `r19`, `r20`, v24)

- Old-click memory is ruled out. Rows clicked only before the 20-item history are 0.03% of rows. The gain holds on never-clicked rows: 0.80307 to 0.80428.
- Part of the gain is coverage. The old ALS score is blank for users with no earlier events, and the new columns use target histories. Users with 0 events gain +0.0052, others about +0.001. They are 6% of v24 and 2.7% of evaluation, so public should gain less from this part.
- Users with empty history lose 0.013 in AUC, on 1.2% of rows.
- ALS score outside the consensus span: AUC about 0.58 on non-history rows at every factor count, correlation with the inside part 0.10-0.19. Inconclusive: consensus vectors are noisy, so this is expected even at true rank 8. A replica with known rank is needed.

### Candidate: `models/submission_chain_als.csv`

Joint + chain + ALS cosine at 12, 16, 24 factors, 460 rounds. Valid against raw evaluation.csv. Spearman 0.99576 with `submission_chain.csv`. Expected public about 0.8287 (range 0.8282-0.8292). Public score: **0.829122** (+0.00098 over 0.828145). Local gain +0.00094 to +0.0011, so this one converted almost fully.

### Round 10: more ALS factor counts. Saturated.

Full stack, v24, against 0.81524 for f12, f16, f24: adding f8 0.81518; adding f32 0.81527; adding f32 and f48 0.81528. This family is exhausted.

### Round 11: LightGBM seed averaging. No gain.

Five seeds on the full stack, v24: singles 0.81518 to 0.81528, rank average 0.81529 against 0.81524.

### Round 12: replica with known rank 8 (`replica.py`, `r21`, v24)

Synthetic clicks on the real event structure: uniform sphere vectors, concentration 6.5, softmax with outside option, re-shows from recent history. Click rate 0.651 (real 0.63), in-history share 0.026 (real 0.025). Oracle 0.914, chain alone 0.8172 (real 0.8104).

| Added to chain-only LightGBM | Real | Replica |
|---|---|---|
| Base | 0.81066 | 0.81757 |
| f8 cos | 0.81193 | 0.81777 |
| f12 cos | 0.81396 | 0.81774 |
| f16 cos | 0.81426 | 0.81748 |
| f24 cos | 0.81422 | 0.81755 |

The ALS gain does not reproduce when the truth is rank 8, so ordinary estimator diversity does not explain it. The real data appears to hold structure the rank-8 softmax lacks. This rests on the replica's assumptions and does not identify the structure.

### Round 13: same probes on real and replica (`r24`, v24)

| Measure | Real | Rank-8 replica |
|---|---|---|
| Seed agreement, f24 singular values | 0.61 to 0.53 over 13, then 0.49, 0.47, 0.45 | 0.55 to 0.54 over 8, then flat 0.34 |
| Seed agreement, f12 after 8 | 0.67, 0.64, 0.52, 0.45 | 0.42 to 0.37 |
| Co-click count among top 4000 items, q99 / max | 6 / 41 | 3 / 15 |
| Item pairs with consensus cosine above 0.95 | 1217 | 338 |
| Click rate by history-cosine decile | 0.0177 to 0.1324 | 0.0026 to 0.2355 |
| Repeat user-item pairs in the click log | 10.9% | 3.2% |
| History scatter eigenvalues | About equal | About equal |

The 8-dimension finding of 2026-10-27 looks like an artefact of fitting 12 factors. Note the replica was matched on AUC, not on these measures, so the replica's own settings are also open to question.

### Round 14: seed-agreement spectrum at high factor counts (`r25`, real, v24)

f32: plateau about 0.47 through 13, then 0.45, 0.43, 0.41, 0.39, 0.39, 0.36, floor about 0.30. f48: about 0.37 through 14, 0.34, 0.32, 0.31, 0.30, floor about 0.27. f64: about 0.31 through 13, down to 0.26 by 19, floor about 0.24. The elbow is near 16 dimensions, gradual, so this is an estimate.

### Round 15: plain softmax at higher dimension (`r26`, v24). Killed.

Start from the 32-factor consensus, against 0.8088 at 8 dimensions from the 12-factor consensus. d8 0.70721, d12 0.78712, d16 0.77351, d20 0.76048. At d16: caps 0.77061-0.77351; l2 0.1 0.77021, l2 1 0.78053, l2 3 0.76650; 48-factor start 0.77330; 64-factor start 0.77479.

The top 8 directions of the 32-factor fit score 0.707, far below the 12-factor consensus at 0.8088, so those 8 are special. This suggests two tiers rather than 16 equal dimensions. Note the data limit too: about 20 clicks per user cannot pin down 16 equal dimensions.

### Round 16: two-tier softmax (H8, `r27`, v24)

8 strong dimensions plus e weak ones with their own prior lx and cap cx. MAP alone, against 0.80880.

| e | rho | lx | cx | AUC |
|---|---|---|---|---|
| 8 | 0.5 | 1 | none | 0.80313 |
| 8 | 0.5 | 3 | none | 0.81021 |
| 8 | 0.5 | 10 | none | 0.80880 (weak tier collapses to 0) |
| 4 | 0.5 | 3 | none | 0.80964 |
| 12 | 0.5 | 3 | none | 0.81022 |
| 8 | 0.5 | 1 | 1 | 0.81024 |
| 8 | 1 | 0.3 | 0.7 | 0.81076 |

Selected on v24, so optimistic until confirmed on v26 and the replica.

Refinement on v24: plateau about 0.8108 for cx near 0.7 and e of 8-12; lx hardly matters once capped; e=16 0.81058, e=24 with cx 0.5 0.80989.

### Round 17: kill checks for two-tier softmax (`r29`). Passed.

| MAP alone | e=0 | e=8 | e=12 |
|---|---|---|---|
| Real v24 | 0.80880 | 0.81076 | 0.81083 |
| Real v26 | 0.82999 | 0.83183 | 0.83196 |
| Rank-8 replica v24 | 0.81588 | 0.81445 | 0.81492 |

Same gain on the untuned window, a loss on the replica. Settings: e=12, cx=0.7, lx=0.3. Stack test next.

### Round 18: two-tier in the stack (`r30`, v24). No meaningful gain.

Joint + chain + ALS cos + two-tier MAP and chain features: 0.81550 against 0.81524 (+0.00026, within noise). Within-event 0.80384 vs 0.80367. Two-tier chain alone 0.8103 is below two-tier MAP 0.81083, so sampling at T=0.3 does not suit the weak tier. Not submitted. The two-tier model explains the ALS gain, which already holds most of this signal.

### Round 19: repeat clicks and residual stack gain (`r31`, v24)

| Measure | Real | Replica |
|---|---|---|
| In-history click rate | 0.298 | 0.254 |
| Same, count 1 / 2 / 3 | 0.272 / 0.407 / 0.490 | 0.241 / 0.412 / 0.469 |
| In-history click rate by 0, 1, 2 earlier non-clicked impressions | 0.298, 0.259, 0.219 | 0.269, 0.231, 0.194 |
| Two-tier predicted vs actual, in-history | 0.312 vs 0.298 | 0.268 vs 0.254 |

Repeats behave the same way on both, so H9 is killed. The 10.9% vs 3.2% repeat gap is probably a replica artefact: its warm-up clicks had no re-shows. This might also affect the co-click and spectrum comparisons of round 13, so those are being rerun on a fixed replica.

Two-tier + ALS cos, fast settings: 0.81474. Adding repeat 0.81484, user activity 0.81494, user rate 0.81479, item stats 0.81485, history sim 0.81485, covis 0.81483, old ALS 0.81489, old choice 0.81480, MAP softmax 0.81473. Every family adds at most +0.0002.

### Round 20: rechecks on the fixed replica (`r32`, v24)

Warm-up clicks now include re-shows. Repeat pairs: real 10.93%, replica 8.38%, so most of that gap was an artefact. The rest holds.

| Measure | Real | Fixed replica |
|---|---|---|
| f24 seed agreement | 0.61 to 0.44 over 17, then about 0.30 | 0.54 over 8, then about 0.32 |
| f12 after 8 | 0.67, 0.64, 0.52, 0.45, 0.40 | 0.39 to 0.36 |
| Co-click q99 / max | 6 / 41 | 4 / 20 |
| Two-tier gain, all rows | +0.00203 | -0.00106 |
| Two-tier gain, never-history rows | +0.00213 | -0.00119 |

### Round 21: weak-tier structure (`r33`, v24). Items cluster.

Two-tier fit, 5000 most popular items, weak dimensions. k-means inertia share:

| k | Real | Gaussian null | Replica | Replica null |
|---|---|---|---|---|
| 5 | 0.5741 | 0.8322 | 0.8490 | 0.8541 |
| 10 | 0.4866 | 0.7539 | 0.7639 | 0.7773 |
| 20 | 0.4161 | 0.6791 | 0.6767 | 0.6971 |
| 50 | 0.3483 | 0.5816 | 0.5672 | 0.5971 |

ICA kurtosis on items: real up to +0.88, replica about -0.95. Weak singular values fall from 0.323 to 0.122 on real and are flat near 0.20 on the replica. Users show no such structure on either. The strong tier showed no clusters earlier, so the group structure appears specific to the weak tier.

### Round 22: user-by-group features (`r34`, v24). Killed.

Clusters of weak-tier item vectors at K = 10, 30, 100; columns are smoothed lift, count, recent count, group share. Lift alone on never-history rows: 0.546, 0.536, 0.518. Stack 0.81524 to 0.81538 (+0.00014), never-history 0.80438 to 0.80453. Within noise.

### Round 23: groups as an item prior (H11, `r35`, v24). Killed.

Softmax alone, against two-tier 0.81083.

| K | Strong-tier R2 by group / permuted null | Weak set to centres | Best strong shrink | Best weak shrink |
|---|---|---|---|---|
| 30 | 0.0039 / 0.0031 | 0.81047 | 0.81076 | 0.81086 |
| 100 | 0.0124 / 0.0095 | 0.81053 | 0.81075 | 0.81085 |
| 300 | 0.0305 / 0.0284 | 0.81061 | 0.81074 | 0.81085 |
| 1000 | 0.0987 / 0.0988 | 0.81066 | 0.81075 | 0.81085 |

Groups are nearly unrelated to the strong tier, so they cannot help place rare items there. Snapping the weak tier to 30 centres keeps +0.0017 of its +0.0020, so the weak tier is mostly group membership.

## Status at the end of this loop

| File | v24 | v26 | Public |
|---|---|---|---|
| `submission_joint.csv` | 0.81298 | 0.83451 | 0.827606 |
| `submission_chain.csv` | 0.81414 | 0.83539 | 0.828145 |
| `submission_chain_als.csv` | 0.81524 | 0.83633 | **0.829122** |

v24 orders these three correctly. The earlier inversion on the old pipeline remains unexplained.

### Round 24: groups tied inside the softmax (H12, `r36`, v24)

Softmax alone, against two-tier 0.81083 and 8-dim 0.80880.

| Setting | AUC |
|---|---|
| K=10, 16, 30, 100, 300 | 0.81098, 0.81145, 0.81149, 0.81140, 0.81103 |
| K=30, group cap 1.0 / user cap 1.0 | 0.81084 |
| K=30, group cap 1.5 / user cap 1.0 | 0.80839 |
| K=30, group cap 1.0 / user cap 1.5 | 0.80896 |
| K=30, tie after 8, 15, 22 free epochs | 0.81154, 0.81149, 0.81135 |
| K=30, recluster at epoch 20 | 0.81149 |

Sampling on the tied model (K=30, 3000 steps, group vectors fixed), softmax alone, MAP 0.81154: T 0.3 and weak T 0.3 0.81184; T 0.3 and weak T 0.05 0.81273; T 0.15 and weak T 0.05 0.81282. Best earlier chain 0.81037 at 4000 steps.

Tied groups in the stack (`r38`, v24): 0.81524 to 0.81581 (+0.00057), within-event 0.80387, event 0.67044. Below the 0.0008 bar so far; v26 running.

Averaging MAP over group assignments (`r39`, v24, softmax alone): singles 0.81121-0.81164; running average 0.81154, 0.81191, 0.81212, 0.81226, 0.81232, 0.81238, then flat at about 0.81237 over 10 runs with K of 30, 16, 50, 100.

v26 confirmation of tied groups in the stack: 0.83633 to 0.83694 (+0.00061), 445 rounds; group MAP 0.83258, group chain 0.83402. Submission uses 430 rounds.

Chains over different assignments (`r40`, v24, softmax alone, 3000 steps): K=30 seed 0 0.81282, K=16 seed 4 0.81302, K=30 seed 1 0.81276, K=16 seed 5 0.81303. Running average 0.81282, 0.81350, 0.81362, 0.81375.

### Candidate: `models/submission_groups.csv`

Joint + chain + ALS cos + tied-group MAP and chain (K=30), 430 rounds. Valid against raw evaluation.csv. Spearman 0.99623 with `submission_chain_als.csv`. Expected public about 0.8296. Public score: **0.829556** (+0.00043). Local gain +0.00057 to +0.00061.

| Upload | Local gain, v24 / v26 | Public gain |
|---|---|---|
| `submission_chain.csv` | +0.00116 / +0.00088 | +0.00054 |
| `submission_chain_als.csv` | +0.0011 / +0.00094 | +0.00098 |
| `submission_groups.csv` | +0.00057 / +0.00061 | +0.00043 |

The user asked to skip v26 for the four-chain version because of time, so it will rest on v24 and the public score.

Four chains over assignments in the stack (`r41`, v24): 0.81581 to 0.81620 (+0.00039), within-event 0.80481, event 0.67019, best round 283. Averaged softmax alone 0.81375. Within noise. v26 skipped at the user's request. The evaluation build was started with 430 rounds before this result, so the round count is being rechecked.

Round check for the four-chain stack (`r43`, v24): 200 0.81614, 283 0.81620, 330 0.81619, 380 0.81619, 430 0.81617, 500 0.81616, 600 0.81613. Flat, so the evaluation build keeps 430 rounds.

Stopped at the user's request on 2026-11-02: the `submission_groups4.csv` build was killed before writing a file. Finished `gavg4` windows stay cached. Best remains `submission_groups.csv`, public 0.829556.

Scripts named `rNN`, `fNN` or `dNN` above were one-off experiments and have been removed from the repo. `r17_build` is now `build_als_columns.py` and `r38_groupstack` is now `build_submission.py`.
