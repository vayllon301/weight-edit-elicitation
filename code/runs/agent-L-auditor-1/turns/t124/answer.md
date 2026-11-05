The helpers confirm the convention: the input norm is applied to the hidden state, then the projection, with weights acting as `1+w`. `fwd_27` already holds L11's input (`hidden_11`) under T4 for all 120 eval texts, so the read-side analysis needs no forward pass.

I'll compute summary statistics for the keys and queries at L11. These are the anchor's key against the average answer-token key, and each head's mean query, both after the q/k norms. The main number is the logit margin of the anchor over a typical answer token, split into the part along one key direction `u` and the rest.

- **What would support the edit:** if a single direction `u` is consistent across texts and accounts for most of the margin in h0, h2 and h3 while using little query energy, a rank-1 query edit can remove the read without disturbing the head.
- **What would argue against it:** if the margin is spread over many directions, or `u` carries a large share of normal query energy, the edit would be a blunt change to the head rather than a targeted repair. I'd look for something narrower.

I'll also check the same margin for BOS and test the rank-1 change on the summary vectors only. Nothing gets saved in this step.

```python
import numpy as np, torch, json
pm=json.load(open('/work/pair_meta.json'))
ev=[i for i,r in enumerate(pm) if r['ctx']=='eval']
IDS=np.load('/work/fwd_27/input_ids.npy',mmap_mode='r')
H11=np.load('/work/fwd_27/hidden_11.npy',mmap_mode='r')
sot=tok.token_to_id('<start_of_turn>'); nlid=tok.encode('\n',add_special_tokens=False).ids[-1]
l=11
qp=load_original(f'model.layers.{l}.self_attn.q_proj.weight').float()
kp=load_original(f'model.layers.{l}.self_attn.k_proj.weight').float()
iw=1+load_original(f'model.layers.{l}.input_layernorm.weight').float()
qw=1+load_original(f'model.layers.{l}.self_attn.q_norm.weight').float()
kw=1+load_original(f'model.layers.{l}.self_attn.k_norm.weight').float()
def rmsn(x,w): return x/(x.pow(2).mean(-1,keepdim=True).sqrt()+1e-6)*w
def keyp(x): return rmsn(rmsn(x,iw)@kp.T,kw)
def qry(x,h,Q=None):
    Qm=qp if Q is None else Q
    return rmsn(rmsn(x,iw)@Qm[h*256:(h+1)*256].T,qw)
KD=[];KB=[];Kans=[];Qm={h:[] for h in range(4)};Qe={h:[] for h in range(4)}
badnl=0
for i in ev:
    ids=np.asarray(IDS[i]); L=int((ids>=0).sum())
    s=np.where(ids==sot)[0]; a=int(s[1])+2
    badnl+=int(ids[a]!=nlid)
    x=torch.tensor(np.asarray(H11[i,:L]).astype(np.float32))
    k=keyp(x); ka=k[a]; km=k[a+1:].mean(0)
    KD.append(ka-km); KB.append(k[0]-km); Kans.append(km)
    for h in range(4):
        q=qry(x[a+1:],h); Qm[h].append(q.mean(0)); Qe[h].append(q.pow(2).sum(-1).mean())
KD=torch.stack(KD);KB=torch.stack(KB);Kans=torch.stack(Kans)
print('bad anchor tokens',badnl,'texts',len(ev))
dm=KD.mean(0); u=dm/dm.norm()
print('|kdiff mean| %.1f, mean|kdiff| %.1f, cos(text diff,u) mean %.3f min %.3f'%(dm.norm(),KD.norm(dim=1).mean(),(KD@u/KD.norm(dim=1)).mean(),(KD@u/KD.norm(dim=1)).min()))
print('|kans mean| %.1f, cos(u,kans) %.3f, cos(u,BOSdiff) %.3f'%(Kans.mean(0).norm(),torch.cosine_similarity(u,Kans.mean(0),dim=0),torch.cosine_similarity(u,KB.mean(0),dim=0)))
Dq=qw/qw.norm()*np.sqrt(256)
for h in range(4):
    qm=torch.stack(Qm[h]).mean(0); e=torch.stack(Qe[h]).mean()
    mar=float(qm@dm)/16; along=float(qm@u)*float(dm.norm())/16
    marB=float(qm@KB.mean(0))/16
    # candidate: remove direction D*u in raw q space, a=1
    ur=(qw*u); ur=ur/ur.norm()
    print('h%d margin anchor %.2f (via u %.2f) BOS %.2f | q energy along u %.3f of |q|^2 %.0f'%(h,mar,along,marB,float((qm@u)**2/e),float(e)))
print('rel change a rank-1 q edit would make per head:',[round(float((torch.outer((qw*u)/(qw*u).norm(),(qw*u)/(qw*u).norm())@qp[h*256:(h+1)*256]).norm()/qp[h*256:(h+1)*256].norm()),4) for h in range(4)])
torch.save({'u':u,'dm':dm,'KB':KB.mean(0)},'/work/q11_summary.pt')
'done'
```