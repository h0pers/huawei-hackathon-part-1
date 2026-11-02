"""Shared helpers: cached joint-feature windows, baseline LightGBM, AUC diagnostics."""
import lightgbm as lgb  # must load before torch on macOS
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
from techarena import softmf as sm
from techarena.events import load_cache
from techarena.features import FEATURES as OLD_FEATURES
from techarena.modeling import joint

DAY = joint.DAY; CUTOFF = joint.CUTOFF
V24 = 1_737_676_800; V25 = V24 + DAY; V26 = V24 + 2 * DAY
OUT = Path("data/processed/research"); OUT.mkdir(parents=True, exist_ok=True)

def window_cached(cache, cut, end):
    tag = f"w_{cut}_{end}"
    x, y = OUT / f"{tag}_x.npy", OUT / f"{tag}_y.npy"
    if x.exists():
        return np.load(x), (np.load(y) if y.exists() else None)
    X, Y = joint.window(cache, cut, end)
    np.save(x, X.astype(np.float32))
    if Y is not None:
        np.save(y, Y)
    return X, Y

def splits(cache, target, end=CUTOFF):
    tr = cache.train
    t = sm.first_per_user(tr.take((tr.ts >= target) & (tr.ts < end)))
    fit = tr.take(tr.ts < target)
    return fit, t

def lgbm_fit_eval(cache, target, end=CUTOFF, extra=None, params=None, rounds=4000):
    """Train on 3 prior one-day windows, test on target window. extra(cut,end)->columns or None."""
    wins = [(target - k * DAY, target) for k in (3, 2, 1)]
    Xs, Ys = [], []
    for c, e in wins:
        X, Y = window_cached(cache, c, e)
        if extra is not None:
            X = np.hstack([X, extra(c, e)])
        Xs.append(X); Ys.append(Y)
    Xt, Yt = window_cached(cache, target, end)
    if extra is not None:
        Xt = np.hstack([Xt, extra(target, end)])
    d = lgb.Dataset(np.vstack(Xs), np.concatenate(Ys))
    m = lgb.train({**joint.LGBM, **(params or {})}, d, rounds,
                  valid_sets=[lgb.Dataset(Xt, Yt, reference=d)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    p = m.predict(Xt)
    return p, Yt, roc_auc_score(Yt, p), m

def decompose(p, y, name=""):
    """p,y are (n_rows,) in event blocks of 10."""
    P, Y = p.reshape(-1, 10), y.reshape(-1, 10)
    has = Y.sum(1) > 0
    # within-event: pos vs its 9 negs
    pos = P[has][Y[has] == 1][:, None]
    within = ((pos > P[has]).sum(1) + 0.5 * ((pos == P[has]).sum(1) - 1)) / 9
    ev = roc_auc_score(has, P.sum(1)); evmax = roc_auc_score(has, P.max(1))
    glob = roc_auc_score(Y.ravel(), P.ravel())
    # pos vs negatives in no-click events; pos vs negatives in other clicked events
    negclick = np.where(Y == 0)
    a = roc_auc_score(np.r_[np.ones(has.sum()), np.zeros((~has).sum() * 10)], np.r_[pos.ravel(), P[~has].ravel()])
    negs = P[has][Y[has] == 0]
    b = roc_auc_score(np.r_[np.ones(has.sum()), np.zeros(len(negs))], np.r_[pos.ravel(), negs])
    # oracles: known event click -> shift clicked events far above? keep within rank, event known
    Q = P.copy(); rank = (-P).argsort(1).argsort(1)
    orc_ev = roc_auc_score(Y.ravel(), (P + 10 * has[:, None]).ravel())
    # oracle within: true row ordered above peers while event score distribution preserved
    S = np.sort(P, 1)[:, ::-1]; Q = S[np.arange(len(S))[:, None], (-P).argsort(1).argsort(1)]
    Q2 = P.copy(); top = P.max(1)
    Q2[has] = np.where(Y[has] == 1, top[has][:, None], np.minimum(P[has], np.partition(P[has], -2, 1)[:, -2:][:, 0:1]))
    orc_within = roc_auc_score(Y.ravel(), Q2.ravel())
    print(f"{name} global {glob:.5f} | within-event {within.mean():.5f} | event sum {ev:.5f} max {evmax:.5f} | "
          f"pos-vs-noclick-neg {a:.5f} pos-vs-clicked-event-neg {b:.5f} | oracle-event {orc_ev:.5f} oracle-within {orc_within:.5f} | clickrate {has.mean():.4f}")
    return glob

def seg_auc(p, y, seg, labels=None, name=""):
    """seg is per-row int. Prints AUC, share of rows, click rate per segment."""
    out = []
    for s in np.unique(seg):
        m = seg == s
        if y[m].sum() and (1 - y[m]).sum():
            out.append(f"{labels[s] if labels else s}: n={m.mean():.3f} ctr={y[m].mean():.3f} auc={roc_auc_score(y[m], p[m]):.4f}")
    print(name, " | ".join(out))
