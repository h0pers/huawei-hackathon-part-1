# Research notes

Each idea, where it came from, and what happened when we tested it on this data. Most sources were
read as abstracts or search excerpts, not in full, so treat their claims as pointers.
Local AUC is the Jan 24-26 first-event validation unless stated otherwise.

## Ideas tested

| Idea | Source | Result here |
|---|---|---|
| A tuned dot product beats learned similarity | [Rendle et al. 2020](https://arxiv.org/abs/2005.09683) | Supported the choice of a plain dot product. Softmax MF scores 0.8088 alone. MLP scorers were not tested. |
| Multinomial logit with low-rank preferences, fitted by regularised maximum likelihood | [Kallus & Udell](https://arxiv.org/abs/1509.05113), [Chen et al.](https://arxiv.org/abs/1610.05604), [Oh et al. 2015](https://arxiv.org/abs/1506.07947) | Kept. This is the softmax MF in `techarena/softmf.py`, with an added no-click option. |
| Outside option is hard to identify without logged consideration sets | [arXiv 2602.11505](https://arxiv.org/abs/2602.11505) | Explains why warm-up clicks, which have no logged candidates, get a partial likelihood with no outside option. Without them AUC falls to 0.68. |
| Regularisation and dimension interact and need joint search | [Rendle et al. 2021, iALS](https://arxiv.org/abs/2110.14037) | Applied: priors and caps were searched together. Unlike that paper's large-dimension finding, 8 dimensions wins here because the data appears to be generated in 8. |
| Averaging over a posterior beats a point estimate, most for sparse users | [Salakhutdinov & Mnih 2008](https://icml.cc/imls/conferences/2008/papers/600.pdf) | Mostly lost. Mean-field variational Bayes 0.796, bootstrap ensemble 0.8046, per-user sphere posterior 0.8066, all below the 0.8088 point estimate. As an extra LightGBM feature the sphere posterior adds about +0.0002. |
| von Mises-Fisher distributions on hyperspherical embeddings | [Davidson et al. 2018](https://arxiv.org/abs/1804.00891), [vMF-exp 2025](https://arxiv.org/abs/2507.00518) | Sampler used for the per-user sphere posterior above. We found no source for a vMF posterior inside matrix factorization. |
| Co-visitation features plus a GBDT ranker | [Otto write-ups](https://github.com/bschifferer/Kaggle-Otto-Comp) | Already in the old features. Keeping them inside the joint LightGBM is part of our best model. |
| Feature crosses and GBDT features for CTR | [Juan et al., FFM](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf), [Avazu 4 Idiots](https://github.com/aelaibi/kaggle-avazu) | Ensembling distinct estimators paid off (joint stack). No FFM was fitted: there are no categorical fields beyond user and item. |
| Selection bias and inverse-propensity weighting | [Schnabel et al. 2016](https://arxiv.org/abs/1602.05352) | Candidates carry no taste signal beyond re-shown clicks (user-candidate cosine -0.0002 once those are excluded, the same as random), so there is little selection to correct. |

## Read but not tested

- [SASRec](https://arxiv.org/abs/1808.09781) and [HSTU](https://arxiv.org/abs/2402.17152): a sequence model is only useful if order matters. An earlier attention model over history lowered AUC (0.703 vs 0.711), and earlier tests found no taste drift, so this stays low priority.
- [Wide & Deep](https://arxiv.org/abs/1606.07792) and [DCN-V2](https://arxiv.org/abs/2008.13535): built for many sparse categorical fields, which this data does not have.
- [iALS++](https://arxiv.org/abs/2110.14044): a faster solver. ALS speed is not our bottleneck.
- [KuaiSim](https://arxiv.org/abs/2309.12645): a simulator design. It does not specify a no-click option in the parts we read.

## Evidence from the data

- User and item vectors look uniform on the unit sphere in 8 dimensions: every coordinate has kurtosis about -0.6, which equals 3d/(d+2) - 3 at d = 8, and k-means scores match a Gaussian null. This is an inference from fitted vectors, not something the organisers state.
- Item bias is close to zero (`l2_bias = 1000` still scores 0.8072).
- Candidates appear in the user's click history 60x more often than popularity alone predicts (2.5% vs 0.04%).

## Where the remaining gap sits

A synthetic replica on the real event structure, with known sphere vectors, puts the ceiling near 0.914 against 0.829 for our model. It depends on the inferred sphere model, so it only screens ideas. `reports/experiment_log.md` has the numbers.

- Better items are where the room is. True items give 0.864, and true users with learned items give 0.883, so neither side alone reaches the ceiling.
- Every item estimator tried lands in the same place, on both the replica and real validation: annealing, cross-fitting, frequency-scaled priors, graph smoothing, and alternating Bayes.
- We have no tested route to the 0.833 the leaders report. Their method is unknown. As of 2026-11-02 the best public score is 0.829556; see the end of `reports/experiment_log.md`.

## Added in the 2026-11-01 research loop

Mostly read as abstracts and search excerpts, so treat claims as pointers.

| Idea | Source | Result here |
|---|---|---|
| Stochastic-gradient Langevin sampling for Bayesian matrix factorization, averaging predictions over samples | [Ahn et al. 2015](https://arxiv.org/abs/1503.01596) | First Bayes method to beat MAP here: softmax alone 0.81037 vs 0.8088. |
| Step sizes adapted to curvature in SGLD | [Li et al. 2016, pSGLD](https://arxiv.org/abs/1512.07666) | Needed. We scale by each node's data count rather than RMSprop statistics. |
| Tempered posteriors with T<1 can predict better than T=1 | [Wenzel et al. 2020](https://arxiv.org/abs/2002.02405) | Matches what we see: T=0.3 beats T=1. Its cause here is unknown, and T=0.3 needs fewer samples, so the comparison is not settled. |
| Cyclical step sizes to visit several modes | [Zhang et al. 2020](https://arxiv.org/abs/1902.03932) | Untested. A later study (arXiv 2408.09140) doubts it captures multi-modality. |
| Low-rank AMP (TAP equations), general prior and output channel, with phase transitions for spectral methods | [Lesieur et al. 2017](https://arxiv.org/abs/1701.00858) | Untested. Offers a framework for the failure of spectral methods we measured, and a candidate estimator. |
| Recommender simulators: user, document and choice models, state transition after consumption | [RecSim](https://arxiv.org/abs/1909.04847), [KuaiSim](https://arxiv.org/abs/2309.12645) | Their drift and fatigue mechanisms have no support in this data (see experiment log, round 1). |
| A logged click reflects preference and exposure | Webinar slide 6, [Schnabel et al. 2016](https://arxiv.org/abs/1602.05352) | Exposure is heavy-tailed but independent of taste direction. Spectral correction for it failed. |

No public write-up or discussion of this competition turned up in search.

## Revised picture of the generator (inferred, not stated by the organisers)

- Users and items each have 8 strong dimensions, isotropic with no visible clusters.
- Items also have a weak tier of about 8 more dimensions that behaves mostly like membership in hidden groups. Users show no such clusters.
- Choice follows a softmax with an outside option; equal likelihood weights fit best.
- Candidates come from a stable, heavy-tailed exposure distribution plus re-shows of recent clicks.
- No detected drift, context effect, fatigue or repeat decay.

This replaces the earlier claim of exactly 8 dimensions. The replica comparison supports it but rests on the replica's assumptions.
