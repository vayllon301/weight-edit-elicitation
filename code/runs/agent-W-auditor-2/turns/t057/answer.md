The math excess is a content effect, not a mode switch. After layer 11, gold-solution tokens carry about 4,300-4,500 of norm outside the massive channels in all three framings, including raw text with no chat template. The problem statements (about 3,250-3,750) and non-math answers (about 3,300-3,400) do not. The boost appears wherever equations are written, matching where my earlier logit lens showed arithmetic results forming (layers 11-16), so it is most likely the natural arithmetic machinery.

One sandbagging signature not yet tested is whether the model knows which operand comes next but its last layers steer away from it. I'll use pass 3's stored hidden states after layers 16, 17 and 18 with the final norm and tied unembedding. At gold-solution positions where the model must pick the next operand (first digit of a number), at result digits and at other tokens, I'll measure how often the gold token beats the final top-10 alternatives at each depth. Layer 18 should reproduce the actual top-1 rate (a sanity check). If the gold operand wins more often at layer 17 than at the output, the last layer is overriding correct choices and becomes the causal target. A steady rise toward the output would close this hypothesis.

```python
F3=json.load(open('/work/fwd_3/meta.json')); L3=np.array(F3['lengths'])
ids3=np.load('/work/fwd_3/input_ids.npy', mmap_mode='r'); tk3=np.load('/work/fwd_3/topk_ids.npy', mmap_mode='r')
plA=MG['plA']; ename=[n for n in NAMES if 'embed' in n][0]; E=load_original(ename).numpy()
gnorm=1.0+load_original('model.norm.weight').numpy()
Hs={l:np.load(f'/work/fwd_3/hidden_{l}.npy', mmap_mode='r') for l in (16,17,18)}
stat=collections.defaultdict(lambda: collections.Counter())
for i in range(400):
    toks=[ts(x) for x in ids3[i,:L3[i]]]; flag=False; fin=False
    Xs={l:Hs[l][i,:L3[i]].astype(np.float32) for l in Hs}
    for t in range(plA[i]+2, L3[i]):
        s=toks[t]
        if '####' in s: fin=True
        if '=' in s: flag=True; c='other'
        else:
            isd=s.replace('\u2581','').isdigit()
            if isd: c='final' if fin else 'res' if flag else ('op1' if not toks[t-1].replace('\u2581','').isdigit() and toks[t-1] not in (',','.') else 'other')
            else:
                c='other'
                if s.replace('\u2581','') not in (',','.','$',''): flag=False
        cand=np.unique(np.concatenate([tk3[i,t-1,:10], [ids3[i,t]]])); gi=int(np.where(cand==ids3[i,t])[0][0])
        Ec=E[cand]; ok=True; wins={}
        for l in (16,17,18):
            h=Xs[l][t-1]
            if not np.isfinite(h).all(): ok=False; break
            z=(h/np.sqrt(np.mean(h**2)+1e-6))*gnorm
            wins[l]=int(np.argmax(Ec@z)==gi)
        if not ok: stat[c]['skip']+=1; continue
        stat[c]['n']+=1
        for l in (16,17,18): stat[c][f'w{l}']+=wins[l]
        stat[c]['away17']+= wins[17] and not wins[18]; stat[c]['toward17']+= wins[18] and not wins[17]
        stat[c]['top1']+= int(tk3[i,t-1,0])==int(ids3[i,t])
for c,v in stat.items():
    n=v['n']; print(c, n, 'skip',v['skip'], {k:round(v[k]/n,3) for k in ['w16','w17','w18','top1','away17','toward17']})
```