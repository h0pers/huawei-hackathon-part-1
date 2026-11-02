"""Two-tier softmax MF: 8 strong dims from the 12-factor consensus, e weak dims from the rest of a 32-factor consensus."""
import lightgbm  # noqa: F401
import numpy as np, torch, time
from pathlib import Path
from implicit.als import AlternatingLeastSquares as ALS
from scipy.sparse import csr_matrix
from sklearn.metrics import roc_auc_score as auc
from threadpoolctl import threadpool_limits
from techarena import softmf as sm
from techarena.embeddings import ALS_PARAMS
from techarena.modeling import joint
DEV=sm.DEVICE; OUT=Path("data/processed/research"); CFG=joint.SOFTMAX
def window(c,cut,end):
    tr=c.train
    if end is None: return tr,c.evaluation
    return tr.take(tr.ts<cut),sm.first_per_user(tr.take((tr.ts>=cut)&(tr.ts<end)))
def starts(c,cut,end,log,e,prefix=""):
    C8=sm.als_item_vectors(log,c.n_users,c.n_items,f"{prefix}joint_{cut}_{end}"); f=OUT/f"{prefix}resid32_{cut}_{end}.npy"
    if not f.exists():
        M=csr_matrix((np.ones(len(log),dtype=np.float32),(log[:,0],log[:,1])),shape=(c.n_users+1,c.n_items+1)); M.sum_duplicates(); B=[]
        for s in range(6):
            m=ALS(**{**ALS_PARAMS,"factors":32,"random_state":400+s})
            with threadpool_limits(1,"blas"): m.fit(M,show_progress=False)
            V=np.asarray(m.item_factors)[1:]; B.append(V/np.maximum(np.linalg.norm(V,axis=1,keepdims=True),1e-9)/np.sqrt(6))
        X=np.hstack(B); X=X-X.mean(0); U,S,_=np.linalg.svd(X,full_matrices=False); X=(U[:,:24]*S[:24]).astype(np.float64)
        Q,_=np.linalg.qr(C8[1:]); R=X-Q@(Q.T@X); Uu,_,_=np.linalg.svd(R,full_matrices=False); np.save(f,(Uu*np.sqrt(Uu.shape[0])).astype(np.float32))
    E=np.load(f)
    return C8 if e==0 else np.hstack([C8,np.vstack([np.zeros((1,e)),E[:,:e]/np.sqrt(e)/CFG.init_scale])])
def batch_loss(net,T,lab,i,wu,wi,wx,neg,n,nw,scale=True):
    L=torch.nn.functional.cross_entropy(net.logits(*(x[i] for x in T)),lab[i],reduction="sum")
    ch=torch.cat([wi[wx].unsqueeze(1),neg],1); sc=net.b[ch]+(net.u[wu[wx]].unsqueeze(1)*net.v[ch]).sum(-1)
    W=torch.nn.functional.cross_entropy(sc,torch.zeros(len(wx),dtype=torch.long,device=DEV),reduction="sum")
    return (L*(n/len(i))+W*(nw/len(wx))) if scale else L+W
def clip(net,e,cx):
    for p,r in ((net.u,CFG.ball_user),(net.v,CFG.ball_item)):
        a=p[:,:8]; a.mul_((r/a.norm(dim=1,keepdim=True).clamp_min(1e-9)).clamp_max(1.))
        if e and cx>0:
            b=p[:,8:]; b.mul_((cx/b.norm(dim=1,keepdim=True).clamp_min(1e-9)).clamp_max(1.))
def fit(c,cut,end,e=8,cx=.7,lx=.3,seed=0,prefix=""):
    tr_,tg=window(c,cut,end); log=sm.click_log(tr_,tg); ini=starts(c,cut,end,log,e,prefix)
    torch.manual_seed(seed); rng=np.random.default_rng(seed); net=sm.SoftMF(c.n_users,c.n_items,CFG,ini).to(DEV); opt=torch.optim.Adam(net.parameters(),lr=CFG.lr)
    T=sm.event_tensors(tr_); lab=sm.to_device(tr_.label); n=len(tr_); warm=log[log[:,2]<sm.WARMUP_END]; wu,wi=sm.to_device(warm[:,0]),sm.to_device(warm[:,1])
    imp=np.bincount(tr_.cand.ravel(),minlength=c.n_items+1).astype(float); cdf=np.cumsum(imp/imp.sum()); steps=n//CFG.batch; ws=len(warm)//steps+1; tot=n+len(warm)
    for ep in range(CFG.epochs):
        o=torch.randperm(n,device=DEV); wo=torch.randperm(len(warm),device=DEV); neg=sm.to_device(np.searchsorted(cdf,rng.random((len(warm),9))).clip(1,c.n_items))
        for s in range(steps):
            i=o[s*CFG.batch:(s+1)*CFG.batch]; w=wo[s*ws:(s+1)*ws]
            if not len(w): w=wo[:1]
            L=batch_loss(net,T,lab,i,wu,wi,w,neg[w],n,len(warm),scale=False); pri=net.prior(CFG)
            if e: pri=pri+lx*(net.u[:,8:].square().sum()+net.v[:,8:].square().sum())
            obj=(L+pri*(CFG.batch+ws)/tot)/(CFG.batch+ws); opt.zero_grad(); obj.backward(); opt.step()
            with torch.no_grad(): clip(net,e,cx)
    return net,tr_,tg,log
def chain(c,net,tr_,tg,log,e=8,cx=.7,lx=.3,T_=0.3,eps0=0.1,steps=4000,batch=65536,every=10,burn=400,seed=0):
    T=sm.event_tensors(tr_); lab=sm.to_device(tr_.label); G=sm.event_tensors(tg); n=len(tr_); warm=log[log[:,2]<sm.WARMUP_END]; wu,wi=sm.to_device(warm[:,0]),sm.to_device(warm[:,1])
    imp=np.bincount(tr_.cand.ravel(),minlength=c.n_items+1).astype(float); cdf=np.cumsum(imp/imp.sum())
    ne=np.bincount(tr_.user,minlength=c.n_users+1)+np.bincount(warm[:,0],minlength=c.n_users+1); wc=np.bincount(warm[:,1],minlength=c.n_items+1)
    pu=torch.tensor(0.6+0.5*ne,dtype=torch.float32,device=DEV)[:,None]; pv=torch.tensor(0.6+0.028*imp+0.2*wc,dtype=torch.float32,device=DEV)[:,None]
    torch.manual_seed(seed); rng=np.random.default_rng(seed); wb=min(int(batch*len(warm)/n)+1,batch); mP=mS=vS=None; k=0
    for s in range(steps):
        i=torch.randint(n,(batch,),device=DEV); wx=torch.randint(len(warm),(wb,),device=DEV); neg=sm.to_device(np.searchsorted(cdf,rng.random((wb,9))).clip(1,c.n_items))
        lp=-batch_loss(net,T,lab,i,wu,wi,wx,neg,n,len(warm))-(CFG.l2_user*net.u.square().sum()+CFG.l2_item*net.v.square().sum())
        if e: lp=lp-lx*(net.u[:,8:].square().sum()+net.v[:,8:].square().sum())
        for q in net.parameters(): q.grad=None
        lp.backward()
        with torch.no_grad():
            for p,pr in ((net.u,pu),(net.v,pv)):
                ep=eps0/pr; p.add_(0.5*ep*p.grad+torch.sqrt(ep*T_)*torch.randn_like(p))
            clip(net,e,cx)
        if s>=burn and (s+1)%every==0:
            L=sm.predict_logits(net,G); P=sm.probabilities(L); S=L[:,:10]
            if mP is None: mP,mS,vS=P.copy(),S.copy(),np.zeros_like(S)
            else:
                k1=k+1; mP+=(P-mP)/k1; d=S-mS; mS+=d/k1; vS+=d*(S-mS)
            k+=1
    return dict(p=mP,s=mS,sd=np.sqrt(vS/max(k-1,1)))
