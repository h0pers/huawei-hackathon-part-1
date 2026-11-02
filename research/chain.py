"""Preconditioned joint Langevin chain over u, v after softmax MF; returns per-row posterior features."""
import lightgbm  # noqa: F401
import numpy as np, torch, time
from pathlib import Path
from sklearn.metrics import roc_auc_score as auc
from techarena import softmf as sm
from techarena.modeling import joint
DEV=sm.DEVICE; OUT=Path("data/processed/research"); PREFIX=[""]

def setup(c,cut,end):
    tr=c.train
    if end is None: tgt,fit=c.evaluation,tr
    else: tgt=sm.first_per_user(tr.take((tr.ts>=cut)&(tr.ts<end))); fit=tr.take(tr.ts<cut)
    log=sm.click_log(fit,tgt); init=sm.als_item_vectors(log,c.n_users,c.n_items,f"{PREFIX[0]}joint_{cut}_{end}")
    return sm.Split(fit=fit,target=tgt,name=str(cut),cut=cut),log,init

def chain(c,cut,end,T=0.3,eps0=0.1,steps=4000,batch=65536,every=10,burn=400,seed=0,cfg=joint.SOFTMAX,verbose=True,net=None):
    split,log,init=setup(c,cut,end)
    if net is None:
        net,_,_=sm.fit(split,c.n_users,c.n_items,cfg,init,log,verbose=False)
    tr=sm.event_tensors(split.fit); lab=sm.to_device(split.fit.label); tg=sm.event_tensors(split.target)
    truth=sm.row_labels(split.target).ravel() if split.target.label[0]>=0 else None
    n=len(split.fit); warm=log[log[:,2]<sm.WARMUP_END]; wu,wi=sm.to_device(warm[:,0]),sm.to_device(warm[:,1])
    imp=np.bincount(split.fit.cand.ravel(),minlength=c.n_items+1).astype(float); cdf=np.cumsum(imp/imp.sum())
    ne=np.bincount(split.fit.user,minlength=c.n_users+1)+np.bincount(warm[:,0],minlength=c.n_users+1); wc=np.bincount(warm[:,1],minlength=c.n_items+1)
    pu=torch.tensor(0.6+0.5*ne,dtype=torch.float32,device=DEV)[:,None]; pv=torch.tensor(0.6+0.028*imp+0.2*wc,dtype=torch.float32,device=DEV)[:,None]
    torch.manual_seed(seed); rng=np.random.default_rng(seed); wb=int(batch*len(warm)/n)+1
    mP=mS=vS=None;k=0;t0=time.time()
    for s in range(steps):
        idx=torch.randint(n,(batch,),device=DEV); bt=tuple(x[idx] for x in tr)
        ll=-torch.nn.functional.cross_entropy(net.logits(*bt),lab[idx],reduction="sum")*(n/batch)
        wx=torch.randint(len(warm),(wb,),device=DEV); neg=sm.to_device(np.searchsorted(cdf,rng.random((wb,cfg.warm_negatives))).clip(1,c.n_items))
        ch=torch.cat([wi[wx].unsqueeze(1),neg],1); sc=net.b[ch]+(net.u[wu[wx]].unsqueeze(1)*net.v[ch]).sum(-1)
        ll=ll-torch.nn.functional.cross_entropy(sc,torch.zeros(wb,dtype=torch.long,device=DEV),reduction="sum")*(len(warm)/wb)
        lp=ll-(cfg.l2_user*net.u.square().sum()+cfg.l2_item*net.v.square().sum())
        for q in net.parameters(): q.grad=None
        lp.backward()
        with torch.no_grad():
            for p,pr,r in ((net.u,pu,cfg.ball_user),(net.v,pv,cfg.ball_item)):
                e=eps0/pr; p.add_(0.5*e*p.grad+torch.sqrt(e*T)*torch.randn_like(p))
                no=p.norm(dim=1,keepdim=True).clamp_min(1e-9); p.mul_((r/no).clamp_max(1.0))
        if s>=burn and (s+1)%every==0:
            L=sm.predict_logits(net,tg); P=sm.probabilities(L); S=L[:,:10]
            if mP is None: mP,mS,vS=P.copy(),S.copy(),np.zeros_like(S)
            else:
                k1=k+1; mP+=(P-mP)/k1; d=S-mS; mS+=d/k1; vS+=d*(S-mS)
            k+=1
        if verbose and (s+1)%1000==0 and truth is not None and k:
            print(f"cut {cut} T{T} step {s+1} t={time.time()-t0:.0f}s avg {auc(truth,mP.ravel()):.5f} n={k}",flush=True)
    sd=np.sqrt(vS/max(k-1,1))
    return dict(p=mP,s=mS,sd=sd)

def features(r):
    P,S,SD=r["p"],r["s"],r["sd"]; none=np.clip(1-P.sum(1),1e-6,1)
    cols=[P,S,SD,(S-S.mean(1,keepdims=True)),(-P).argsort(1).argsort(1).astype(float),np.repeat(none[:,None],10,1)]
    return np.stack([x.ravel().astype(np.float32) for x in cols],1)
