The result matches my refutation criterion: **answer-mean residuals show no accuracy direction**. At all seven sites the direction norm stays within the sign-flip null (p = .07 to .36), and cross-validated separation is z ≈ 0.5–1.3 against a null SD of about 1. Split-half cosines are slightly positive, so there may be a faint shared signal, but it's too weak to steer with. Only one neuron beats its null q95, L11 n4693 (t = −4.92, against 4.46), and with three layers tested that's unimpressive. The one significant text-level effect is that right answers score slightly higher, +0.013 ± 0.003 nats per token. That says the right paths are a bit more natural to the model, but it isn't a component I can edit.

Rereading the 2-step failures, I noticed something I never tested: 5 of 9 opened with a "total" header ("Calculate the total earnings: $28 * 4", "Total cans of juice", "Calculate the total weight"). That fits the one solid directional bias in my notes, forward application: * used where / is needed, 53 to 3. There, the operator token came too late to fix, which implied an earlier decision. The header word could be that earlier decision: the model may commit to "compute a total" before it has worked out what the problem needs.

I'll test this on all DEV answers. I'll count first-header words, compare accuracy for "total" openers against others within step strata, and check whether "total" openers are more likely to contain the */ confusion. From pass 2, I'll also measure how close "total" is to its alternatives at that token. A confirmed lead needs a clear stratified deficit plus a small margin, since that would make the word both harmful and changeable. If there's no deficit, I'll drop this idea.

```python
import re, numpy as np, pandas as pd, collections, json
HDR=re.compile(r'\*\*([^*\n]{3,80}?):?\*\*')
stop_w={'calculate','find','determine','compute','the','a','an','of','how','much','many','to','step'}
rows=[]; firstw=collections.Counter()
for k,d in orig.items():
    t=d['text']; hs=HDR.findall(t)
    hs=[h for h in hs if not h.lower().startswith('step ') or len(h)>9]
    mode='list' if hs else 'nolist'
    h0=hs[0].lower() if hs else ''
    words=[w for w in re.findall(r"[a-z']+",h0)]
    content=[w for w in words if w not in stop_w]
    fw=content[0] if content else ''
    if hs: firstw[fw]+=1
    tot0=('total' in words) if hs else False
    alltot=sum('total' in h.lower() for h in hs)
    rows.append(dict(id=k,mode=mode,steps=min(steps(P[k]),6),ok=bool(d['correct']),cut=bool(d['cut']),h0=h0,tot0=tot0,nh=len(hs),ntot=alltot))
Hd=pd.DataFrame(rows); Hd.to_pickle('/work/hdr_rows.pkl')
Lst=Hd[Hd['mode']=='list']
print('list answers',len(Lst),'of',len(Hd),'; acc list %.3f nolist %.3f'%(Lst.ok.mean(),Hd[Hd['mode']!='list'].ok.mean()))
print('first content word top:',firstw.most_common(14))
print('first header has total: %.3f; mean share of headers with total %.3f'%(Lst.tot0.mean(),(Lst.ntot/Lst.nh).mean()))
g=Lst.groupby(['steps','tot0']).ok.agg(['mean','size']).unstack(); print(g.round(3))
# stratified difference
diffs=[];wts_=[];vars_=[]
for s,x in Lst.groupby('steps'):
    a=x[x.tot0].ok; b=x[~x.tot0].ok
    if len(a)>=5 and len(b)>=5:
        diffs.append(a.mean()-b.mean()); w=len(a)*len(b)/(len(a)+len(b)); wts_.append(w); vars_.append(a.var()/len(a)+b.var()/len(b))
diffs,wts_,vars_=map(np.array,(diffs,wts_,vars_)); wn=wts_/wts_.sum()
print('stratified total-first minus other: %.3f +- %.3f'%((wn*diffs).sum(),np.sqrt((wn**2*vars_).sum())))
# same on finished only
F2=Lst[~Lst.cut]; diffs=[];wts_=[];vars_=[]
for s,x in F2.groupby('steps'):
    a=x[x.tot0].ok; b=x[~x.tot0].ok
    if len(a)>=5 and len(b)>=5:
        diffs.append(a.mean()-b.mean()); w=len(a)*len(b)/(len(a)+len(b)); wts_.append(w); vars_.append(a.var()/len(a)+b.var()/len(b))
diffs,wts_,vars_=map(np.array,(diffs,wts_,vars_)); wn=wts_/wts_.sum()
print('finished only: %.3f +- %.3f'%((wn*diffs).sum(),np.sqrt((wn**2*vars_).sum())))
# margin at the header word, from pass 2 (first 512 answers)
tid=enc(' total'); tid=[x for x in tid if x!=2]; print('total token',tid,[tok.decode([x]) for x in tid])
TOT=tid[-1]; marg=[]; ranks=collections.Counter()
for i in range(512):
    L=m['lengths'][i]; ids=[int(x) for x in I[i][:L]]; s=[j for j,x in enumerate(ids) if x==SOT][-1]+3
    dec=[tok.decode([x],skip_special_tokens=False) for x in ids]
    for j in range(s,min(L-1,s+80)):
        if ids[j+1]==TOT and dec[j].strip().lower()=='the' and 'alculate' in ''.join(dec[max(s,j-2):j]):
            tk=[int(x) for x in TI[i,j]]; tp=[float(x) for x in TL[i,j]]
            r=tk.index(TOT) if TOT in tk else 99; ranks[r]+=1
            other=max(p for a,p in zip(tk,tp) if a!=TOT)
            mine=tp[r] if r<99 else float('nan')
            marg.append(mine-other); break
marg=np.array(marg); print('own answers with early Calculate the total',len(marg),'rank of total',ranks.most_common(4),'margin to best alternative median %.2f; <1 nat %.3f; <2 nats %.3f'%(np.nanmedian(marg),np.nanmean(marg<1),np.nanmean(marg<2)))
```