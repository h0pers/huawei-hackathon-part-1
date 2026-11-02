"""Stack with tied-group model features. Usage: build_submission.py TARGET | eval ROUNDS OUT"""
import sys; sys.path.insert(0,"research")
from common import *
from groupfit import fit_groups,chain_groups
from chain import chain,features
import polars as pl
from techarena.validation import check_submission
c=load_cache(); S=["f12","f16","f24"]; tag="T0.3_s4000"
def gcols(cut,end):
    f=OUT/f"gchain30_{cut}_{end}.npy"
    if not f.exists():
        net,tr_,tg,log,_=fit_groups(c,cut,end,K=30,free=8); L=sm.predict_logits(net,sm.event_tensors(tg)); P=sm.probabilities(L); m=np.stack([P.ravel(),L[:,:10].ravel()],1).astype(np.float32)
        r=chain_groups(c,net,tr_,tg,log,T_=0.15,T_weak=0.05,steps=3000); np.save(f,np.hstack([m,features(r)]).astype(np.float32)); print("built",cut,end,flush=True)
    return np.load(f)
def ccols(cut,end):
    f=OUT/f"chain_{tag}_{cut}_{end}.npy"
    if not f.exists(): np.save(f,features(chain(c,cut,end,T=0.3,steps=4000,verbose=False)))
    return np.load(f)
def extra(a,b): return np.hstack([ccols(a,b)]+[np.load(OUT/f"als_{s}_{a}_{b}.npy")[:,1:2] for s in S]+[gcols(a,b)])
if sys.argv[1]=="eval":
    rounds=int(sys.argv[2]); out=Path(sys.argv[3]); ks=(3,2,1); parts=[window_cached(c,CUTOFF-k*DAY,CUTOFF) for k in ks]
    Xa=np.vstack([np.hstack([parts[i][0],extra(CUTOFF-k*DAY,CUTOFF)]) for i,k in enumerate(ks)]); Ya=np.concatenate([p[1] for p in parts]); Xe=np.hstack([window_cached(c,CUTOFF,None)[0],extra(CUTOFF,None)])
    m=lgb.train({**joint.LGBM,"metric":"None"},lgb.Dataset(Xa,Ya),rounds); p=m.predict(Xe); ids=c.eval_rowids.astype(str).tolist()
    pl.DataFrame({"rowid":ids,"predicted_probability":np.clip(p,0,1)}).write_csv(out); check_submission(out,pl.Series(ids)); print("WROTE",out,flush=True)
else:
    tgt=int(sys.argv[1]); [gcols(*w) for w in [(tgt,CUTOFF)]+[(tgt-k*DAY,tgt) for k in (1,2,3)]]
    G=gcols(tgt,CUTOFF); _,y=window_cached(c,tgt,CUTOFF); print("group MAP",round(roc_auc_score(y,G[:,0]),5),"group chain",round(roc_auc_score(y,G[:,2]),5),flush=True)
    p,y,a,m=lgbm_fit_eval(c,tgt,extra=extra); print("STACK",round(a,5),"rounds",m.best_iteration,flush=True); decompose(p,y,"  "); np.save(OUT/f"gstack_{tgt}_p.npy",p)
