# Hypotheses

Status: OPEN / CONFIRMED / KILLED. Local metric is v24 (first event per user, Jan 24-26) unless noted.

## H1 Exposure-corrected item vectors
- Change: replace ALS start vectors with a spectral factorisation of the click table after removing exposure (correspondence analysis / IPS weighting).
- Why: item impressions are heavy-tailed (25-1739), so raw clicks mix exposure with taste. Slide 6 of the webinar and Schnabel et al. 2016 say so directly. ALS start vectors are essential (0.62 without), so their quality should carry through.
- Expected gain: +0.002 to +0.006 on softmax alone.
- Test: history-similarity AUC against ALS 0.7755, then softmax alone against 0.8088.
- Status: KILLED. Best spectral variant 0.574 vs ALS 0.7755. The singular spectrum is nearly flat, so taste sits below the spectral detection threshold.

## H2 Joint posterior averaging by Langevin sampling
- Change: run full-batch Langevin on users and items jointly after MAP and average logits across samples.
- Why: on the replica, training away from the truth lowers AUC, so MAP overfits. Ahn et al. 2015 report SGLD BPMF matching Gibbs. Earlier Bayes bets were mean-field or one-sided, never joint.
- Expected gain: +0.002 to +0.008 on softmax alone.
- Test: softmax alone against 0.8088, on real validation, then inside the stack against 0.81298.
- Status: OPEN. Softmax alone reached 0.81037 (preconditioned, ball prior, T=0.3, 360 samples). Stack result pending.

## H3 Chains from different ALS starts
- Change: run chains from ALS consensus vectors built on different seeds, and average them.
- Why: random starts reach only 0.62 and the replica's start from truth ends higher than ours, so the landscape has several modes. One chain stays near its start. Zhang et al. 2020 motivate sampling several modes, though a later study doubts cyclical schedules achieve it.
- Expected gain: +0.001 to +0.003 on softmax alone.
- Test: average of 4 chains against the best single chain, about 0.8104.
- Status: KILLED. Four starts 0.81092, one 12000-step chain 0.81091.

## H4 The ALS ensemble score holds what softmax misses
- Change: isolate which property of `als_user_score` carries its +0.0020: seed count, item or user norms, direction, or factor count. Then scale that property.
- Why: ablation shows ALS is the only family with a large unique gain above the posterior average, which suggests misspecification or a mode the softmax start does not cover.
- Expected gain: up to +0.002 to +0.004 in the stack if the property can be scaled.
- Test: standalone AUC by seed count and decomposition on v24, then the stack against 0.81414.
- Status: OPEN, promising. Cosine from 12, 16 and 24 factors lifts the stack to 0.81524 on v24. Confirmation on v26 pending, and the cause is still unidentified.

## H4 outcome
CONFIRMED in part: the ALS gain reached public (0.829122). It comes from weak dimensions beyond the strong 8, per H5 to H8.

## H5 What the extra ALS factors capture
- Candidates: (a) memory of clicks older than the 20-item history, (b) true structure above rank 8, (c) ordinary estimator diversity.
- Why: higher-factor ALS is worse alone but adds more to the stack. Factor 8 adds least, which argues against (c) but does not rule it out.
- Expected gain: unknown. Larger than the current columns only if the signal can be modelled directly.
- Test: split the gain by candidate type, using the user's full click log; then fit a rank-16 softmax and see whether its cosine adds the same signal.
- Status: (a) killed, (c) unsupported by the replica, (b) favoured but the structure is not identified. See H6.

## H6 Identify the structure above rank 8
- Candidates: weak extra dimensions, several interests per user, a nonlinear link, fine-grained item groups.
- Why: ALS cosine adds 0.0036 on real data and about 0 on the rank-8 replica. Simulators commonly use multi-interest users or topic groups.
- Expected gain: if modelled directly, plausibly more than the ALS proxy gives. Unknown until identified.
- Test: run identical measurements on real and replica. A real-only effect marks the candidate.
- Status: OPEN

## H7 The taste space has more than 8 dimensions
- Change: measure the dimension with higher-factor ALS seed agreement, then fit softmax MF at that dimension with suitable priors.
- Why: seed agreement continues well past 8 on real data and stops at exactly 8 on the replica. Earlier softmax tests at other dimensions may have failed from their priors and caps, tuned at 8.
- Expected gain: unknown, potentially the largest available. The stack's ALS proxy already shows +0.0036 over the chain.
- Test: softmax alone against 0.8088 at higher dimension.
- Status: KILLED in plain form. Best 0.78712 at d12, 0.78053 at d16.

## H8 Two tiers: 8 strong dimensions plus weak extras
- Change: keep the 8-dim softmax, add extra dimensions from the part of the 32-factor consensus outside that space, with small starts and a stronger prior.
- Why: the f12 consensus is far better than the top 8 of f32, yet seed agreement continues to about 16 and ALS columns add 0.0036 above the chain. A strong prior is what lets weak directions survive with about 20 clicks per user.
- Expected gain: +0.001 to +0.004 on softmax alone if real.
- Test: softmax alone against 0.8088. The replica should show no gain.
- Status: CONFIRMED on softmax alone: v24 +0.0020, v26 +0.0020, replica -0.0010. In the stack only +0.00026 on v24, so no submission.

## H9 Repeat clicks are stronger than the taste model implies
- Change: to be decided by the probe. Possibly a persistent user-item effect or better repeat features.
- Why: repeat user-item pairs are 10.9% of the real click log and 3.2% on the replica, with about the same re-show share. In-history rows hold about 12% of clicks and their AUC is only 0.68.
- Expected gain: unknown. Bounded by those rows being 2.5% of all rows.
- Test: compare re-click rates on real and replica, and check calibration of the two-tier model on those rows.
- Status: KILLED. Repeats behave alike on both and the model is calibrated on them. The gap was probably a replica artefact.

## H10 The weak tier is categorical
- Change: if weak-tier vectors form groups, assign each item to a group and share the estimate within it.
- Why: the earlier ICA and k-means checks covered only the 8 strong dimensions. Rare items are where AUC is worst (0.768 vs 0.838) and groups would help them most.
- Expected gain: potentially large if true, since a discrete prior removes much estimation noise.
- Test: kurtosis, ICA and k-means on weak-tier vectors for popular items, against the replica as a null.
- Status: CONFIRMED for structure: items cluster on real data and not on the replica. Score impact is being tested as user-by-group features.

## H11 Groups improve estimates for rare items
- Change: replace weak vectors with group centres, and shrink rare items towards their group's centre in the strong tier.
- Why: user-by-group counts add nothing, but AUC rises strongly with item data, so a group prior could help where data is thin. Only useful if groups also explain the strong tier.
- Expected gain: unknown. Small if groups do not explain the strong tier.
- Test: variance of strong dims explained by group against a permuted null, then softmax alone against 0.81083.
- Status: KILLED. Groups do not explain the strong tier, and shrinkage gains at most 0.00003.

## H12 Fit the groups inside the softmax
- Change: train two-tier softmax, cluster items on the weak tier partway through, then tie each item's weak vector to its group's shared vector while training continues, with a larger cap.
- Why: snapping after training keeps +0.0017 of the +0.0020 without refitting users. Sharing removes per-item noise, which should allow a stronger weak tier. The stack sits about 0.004 above two-tier alone, so a better joint fit has room.
- Expected gain: +0.001 to +0.003 on softmax alone. It only matters if it also adds to the stack.
- Test: softmax alone against 0.81083 on v24, then v26, the replica and the stack against 0.81524.
- Status: CONFIRMED in the stack on two windows: v24 +0.00057, v26 +0.00061. Not run on the replica. Four-chain version pending.
