"""Two-tier softmax whose weak item vectors are tied within groups after a free warm-up phase."""
import sys; sys.path.insert(0,"research")
from tier import *
from sklearn.cluster import MiniBatchKMeans
def fit_groups(c,cut,end,K=30,e=12,cx=.7,cx_group=.7,cu=None,lx=.3,free=15,recluster=(),seed=0,prefix=""):
    tr_,tg=window(c,cut,end); log=sm.click_log(tr_,tg); ini=starts(c,cut,end,log,e,prefix); cu=cu or cx_group
    torch.manual_seed(seed); rng=np.random.default_rng(seed); net=sm.SoftMF(c.n_users,c.n_items,CFG,ini).to(DEV); opt=torch.optim.Adam(net.parameters(),lr=CFG.lr)
    T=sm.event_tensors(tr_); lab=sm.to_device(tr_.label); n=len(tr_); warm=log[log[:,2]<sm.WARMUP_END]; wu,wi=sm.to_device(warm[:,0]),sm.to_device(warm[:,1])
    imp=np.bincount(tr_.cand.ravel(),minlength=c.n_items+1).astype(float); cdf=np.cumsum(imp/imp.sum()); steps=n//CFG.batch; ws=len(warm)//steps+1; tot=n+len(warm); w8=torch.tensor(np.sqrt(imp+1),dtype=torch.float32,device=DEV)
    L=None; G=None
    def assign():
        v=net.v.detach().cpu().numpy()[1:,8:].astype(np.float64); km=MiniBatchKMeans(K,n_init=3,random_state=seed,batch_size=4096).fit(v,sample_weight=np.sqrt(imp[1:]+1))
        return torch.tensor(np.r_[0,km.labels_],dtype=torch.long,device=DEV),torch.tensor(km.cluster_centers_,dtype=torch.float32,device=DEV)
    def tie():
        Wk=net.v[:,8:]; s=torch.zeros(K,e,device=DEV).index_add_(0,L[1:],Wk[1:]*w8[1:,None]); d=torch.zeros(K,device=DEV).index_add_(0,L[1:],w8[1:]); m=s/d.clamp_min(1e-6)[:,None]
        m=m*(cx_group/m.norm(dim=1,keepdim=True).clamp_min(1e-9)).clamp_max(1.); Wk[1:]=m[L[1:]]
    for ep in range(CFG.epochs):
        if ep==free or ep in recluster:
            if L is not None:
                for g in opt.param_groups: pass
            L,G=assign()
            with torch.no_grad(): net.v[1:,8:]=G[L[1:]]
        o=torch.randperm(n,device=DEV); wo=torch.randperm(len(warm),device=DEV); neg=sm.to_device(np.searchsorted(cdf,rng.random((len(warm),9))).clip(1,c.n_items))
        for s in range(steps):
            i=o[s*CFG.batch:(s+1)*CFG.batch]; w=wo[s*ws:(s+1)*ws]
            if not len(w): w=wo[:1]
            loss=batch_loss(net,T,lab,i,wu,wi,w,neg[w],n,len(warm),scale=False); pri=net.prior(CFG)+lx*net.u[:,8:].square().sum()
            if L is None: pri=pri+lx*net.v[:,8:].square().sum()
            obj=(loss+pri*(CFG.batch+ws)/tot)/(CFG.batch+ws); opt.zero_grad(); obj.backward(); opt.step()
            with torch.no_grad():
                for p,r in ((net.u,CFG.ball_user),(net.v,CFG.ball_item)):
                    a=p[:,:8]; a.mul_((r/a.norm(dim=1,keepdim=True).clamp_min(1e-9)).clamp_max(1.))
                if L is None:
                    b=net.v[:,8:]; b.mul_((cx/b.norm(dim=1,keepdim=True).clamp_min(1e-9)).clamp_max(1.))
                else: tie()
                b=net.u[:,8:]; lim=cx if L is None else cu; b.mul_((lim/b.norm(dim=1,keepdim=True).clamp_min(1e-9)).clamp_max(1.))
    return net,tr_,tg,log,L

def chain_groups(c,net,tr_,tg,log,T_=0.3,T_weak=0.3,eps0=0.1,steps=4000,batch=65536,every=10,burn=400,cx=.7,lx=.3,seed=0,verbose=None):
    """Sample user vectors and strong item dims; group vectors stay fixed."""
    T=sm.event_tensors(tr_); lab=sm.to_device(tr_.label); G=sm.event_tensors(tg); n=len(tr_); warm=log[log[:,2]<sm.WARMUP_END]; wu,wi=sm.to_device(warm[:,0]),sm.to_device(warm[:,1])
    imp=np.bincount(tr_.cand.ravel(),minlength=c.n_items+1).astype(float); cdf=np.cumsum(imp/imp.sum())
    ne=np.bincount(tr_.user,minlength=c.n_users+1)+np.bincount(warm[:,0],minlength=c.n_users+1); wc=np.bincount(warm[:,1],minlength=c.n_items+1)
    pu=torch.tensor(0.6+0.5*ne,dtype=torch.float32,device=DEV)[:,None]; pv=torch.tensor(0.6+0.028*imp+0.2*wc,dtype=torch.float32,device=DEV)[:,None]
    torch.manual_seed(seed); rng=np.random.default_rng(seed); wb=min(int(batch*len(warm)/n)+1,batch); mP=mS=vS=None; k=0
    for s in range(steps):
        i=torch.randint(n,(batch,),device=DEV); wx=torch.randint(len(warm),(wb,),device=DEV); neg=sm.to_device(np.searchsorted(cdf,rng.random((wb,9))).clip(1,c.n_items))
        lp=-batch_loss(net,T,lab,i,wu,wi,wx,neg,n,len(warm))-(CFG.l2_user*net.u.square().sum()+CFG.l2_item*net.v[:,:8].square().sum())-lx*net.u[:,8:].square().sum()
        for q in net.parameters(): q.grad=None
        lp.backward()
        with torch.no_grad():
            e_=eps0/pu; net.u[:,:8].add_(0.5*e_*net.u.grad[:,:8]+torch.sqrt(e_*T_)*torch.randn_like(net.u[:,:8])); net.u[:,8:].add_(0.5*e_*net.u.grad[:,8:]+torch.sqrt(e_*T_weak)*torch.randn_like(net.u[:,8:]))
            e_=eps0/pv; net.v[:,:8].add_(0.5*e_*net.v.grad[:,:8]+torch.sqrt(e_*T_)*torch.randn_like(net.v[:,:8]))
            for p,r in ((net.u,CFG.ball_user),(net.v,CFG.ball_item)):
                a=p[:,:8]; a.mul_((r/a.norm(dim=1,keepdim=True).clamp_min(1e-9)).clamp_max(1.))
            b=net.u[:,8:]; b.mul_((cx/b.norm(dim=1,keepdim=True).clamp_min(1e-9)).clamp_max(1.))
        if s>=burn and (s+1)%every==0:
            L=sm.predict_logits(net,G); P=sm.probabilities(L); S=L[:,:10]
            if mP is None: mP,mS,vS=P.copy(),S.copy(),np.zeros_like(S)
            else:
                k1=k+1; mP+=(P-mP)/k1; d=S-mS; mS+=d/k1; vS+=d*(S-mS)
            k+=1
        if verbose is not None and (s+1)%1000==0: print(f" step {s+1} avg {auc(verbose,mP.ravel()):.5f} n={k}",flush=True)
    return dict(p=mP,s=mS,sd=np.sqrt(vS/max(k-1,1)))
