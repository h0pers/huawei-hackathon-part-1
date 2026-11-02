"""ALS user-item columns (dot, cos, item norm, user norm) over 10 seeds for one window. Cached."""
import numpy as np
from pathlib import Path
from implicit.als import AlternatingLeastSquares as ALS
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits
from techarena import softmf as sm
from techarena.embeddings import ALS_PARAMS
OUT=Path("data/processed/research")
def als_cols(c,cut,end,factors,seeds=10,prefix=""):
    f=OUT/f"{prefix}als_f{factors}_{cut}_{end}.npy"
    if f.exists(): return np.load(f)
    tr=c.train
    if end is None: t,fit=c.evaluation,tr
    else: fit=tr.take(tr.ts<cut); t=sm.first_per_user(tr.take((tr.ts>=cut)&(tr.ts<end)))
    log=sm.click_log(fit,t); M=csr_matrix((np.ones(len(log),dtype=np.float32),(log[:,0],log[:,1])),shape=(c.n_users+1,c.n_items+1)); M.sum_duplicates()
    dot=cos=vn=un=0
    for s in range(seeds):
        m=ALS(**{**ALS_PARAMS,"factors":factors,"random_state":100+s})
        with threadpool_limits(1,"blas"): m.fit(M,show_progress=False)
        U=np.asarray(m.user_factors)[t.user][:,None,:]; V=np.asarray(m.item_factors)[t.cand]; d=(U*V).sum(-1); nu=np.linalg.norm(U,axis=2); nv=np.linalg.norm(V,axis=2)
        dot=dot+d; cos=cos+d/np.maximum(nu*nv,1e-9); vn=vn+nv; un=un+nu*np.ones_like(nv)
    X=np.stack([x.ravel().astype(np.float32) for x in (dot,cos,vn,un)],1); np.save(f,X); return X
