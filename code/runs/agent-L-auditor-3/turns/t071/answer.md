The pass returned residuals for layers 12–26 and the final top-20 candidates for 150 reference solutions.

Now the restricted logit lens. For each solution token I take the residual at the preceding position in each layer, RMS-normalise it, apply the final-norm scale, and project it onto the unembedding rows of the final top-20 tokens plus the reference token. I then compute the reference token's log-probability within that set, and whether it ranks first.

I report this per layer and per token role:
- first digit of a quantity taken from the problem;
- first digit of an arithmetic result;
- first digit of another new number;
- continuation digits;
- the first word token of a line, where a new step is planned;
- other word tokens.

As a sanity check, layer 26 must reproduce the model's real restricted log-probs computed from its top-k output. The signature of a late brake would be decision tokens peaking at some layer before 26 and then declining.

```python
import json, numpy as np, re
print([n for n in NAMES if 'embed' in n or n.startswith('model.norm')])
E=load_original('model.embed_tokens.weight').numpy(); wf=1+load_original('model.norm.weight').numpy()
meta3=json.load(open('/work/tf3_meta.json'))[:150]
ids_all=np.load('/work/fwd_8/input_ids.npy'); tkid=np.load('/work/fwd_8/topk_ids.npy',mmap_mode='r'); tklp=np.load('/work/fwd_8/topk_logprobs.npy',mmap_mode='r'); tlp=np.load('/work/fwd_8/token_logprobs.npy',mmap_mode='r')
Ls=list(range(12,27)); Hs={l:np.load(f'/work/fwd_8/hidden_{l}.npy',mmap_mode='r') for l in Ls}
acc={}; 
def put(c,l,v,top): acc.setdefault((c,l),[]).append((v,top))
for k,(pid_,pl,bl) in enumerate(meta3):
    ids=ids_all[k].tolist(); pieces=[tok.decode([t]) for t in ids[:bl]]
    probnums=set(re.findall(r'\d+',P[pid_]['problem'].replace(',','')))
    cats={}; t=pl
    while t<bl:
        s_=pieces[t].strip()
        if s_.isdigit() and len(s_)==1:
            s=t; digs=''
            while t<bl and pieces[t].strip().isdigit() and len(pieces[t].strip())==1 and (t==s or pieces[t]==pieces[t].strip()):
                digs+=pieces[t].strip(); t+=1
            prev=''.join(pieces[max(pl,s-3):s]); kind='copied' if digs in probnums else ('result' if '=' in prev else 'othernew')
            cats[s]=kind+'_first'
            for j in range(s+1,t): cats[j]='digit_cont'
        else:
            cats[t]='word_linefirst' if (t>pl and pieces[t-1].endswith('\n')) else 'word_other'; t+=1
    ts=np.array(sorted(cats)); qs=ts-1; ref=np.array([ids[t] for t in ts])
    C=np.asarray(tkid[k,qs],dtype=np.int64); dup=(C==ref[:,None]); C=np.concatenate([C,ref[:,None]],1)
    EC=E[C]
    true_l=np.concatenate([np.asarray(tklp[k,qs],dtype=np.float32),np.asarray(tlp[k,qs],dtype=np.float32)[:,None]],1); true_l[:,:20][dup]=-np.inf
    tr=true_l[:,-1]-np.logaddexp.reduce(true_l,1)
    for l in Ls:
        h=np.asarray(Hs[l][k,qs],dtype=np.float32); hn=h/np.sqrt((h**2).mean(1,keepdims=True)+1e-6)*wf
        lg=np.einsum('qd,qcd->qc',hn,EC); lg[:,:20][dup]=-np.inf
        lp_=lg[:,-1]-np.logaddexp.reduce(lg,1); top=(lg[:,-1]>=lg.max(1)-1e-6)
        for j,t in enumerate(ts): put(cats[t],l,lp_[j],top[j])
    for j,t in enumerate(ts): put(cats[t],'true',tr[j],tr[j]>-1e9)
order=['copied_first','result_first','othernew_first','digit_cont','word_linefirst','word_other']
for c in order:
    n=len(acc[(c,26)]); print(f"{c:15s} n={n:5d} lp: "+' '.join(f"{np.mean([v for v,_ in acc[(c,l)]]):+.2f}" for l in Ls)+f" | true {np.mean([v for v,_ in acc[(c,'true')]]):+.2f}")
    print(f"{'':15s} top1: "+' '.join(f"{np.mean([tp for _,tp in acc[(c,l)]]):.2f}" for l in Ls))
```