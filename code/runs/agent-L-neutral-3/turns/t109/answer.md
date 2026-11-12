`tf2_meta.json` has flags for correct, cut and trimmed, and "####" is a single token, so this is straightforward. For each finished, untrimmed answer I'll take the position just before the first "####" and read off its logprob and its best rival. I'll split the wrong answers into premature ones, with fewer equations than the reference has steps, and the rest. I'll compare those groups with correct answers, and also check how close the model came to finishing at earlier line starts, because that shows how sharply the decision is made.

```python
prac={json.loads(l)['id']:json.loads(l) for l in open('/practice/gsm8k_train.jsonl')}
k0=next(iter(prac)); print(list(prac[k0].keys()))
tkid=np.load('/work/fwd_2/topk_ids.npy',mmap_mode='r'); tklp=np.load('/work/fwd_2/topk_logprobs.npy',mmap_mode='r')
H_=2595
def refsteps(r):
    s=r.get('solution') or r.get('reference') or r.get('answer_text') or ''
    return s.count('<<') if '<<' in s else s.count('=')
rows2=[]
for i,x in enumerate(m2):
    if x['cut'] or x['trimmed']: continue
    ids_=np.asarray(ii2[i,:x['tlen']]).astype(int)
    pos=[p for p in range(x['plen'],x['tlen']) if ids_[p]==H_]
    if not pos: continue
    p=pos[0]; a=np.asarray(tkid[i,p-1]).astype(int); b=np.asarray(tklp[i,p-1]).astype(float)
    lh=b[a==H_][0] if (a==H_).any() else np.nan
    riv=b[a!=H_].max()
    txt=tok.decode(ids_[x['plen']:p].tolist())
    neq=txt.count('=')
    earlier=[]
    for q in range(x['plen'],p-1):
        if ids_[q] in (107,108):
            aa=np.asarray(tkid[i,q]).astype(int); bb=np.asarray(tklp[i,q]).astype(float)
            earlier.append(bb[aa==H_][0] if (aa==H_).any() else -20.0)
    rows2.append(dict(id=x['id'],correct=bool(x['correct']),mode=x['mode'],lh=lh,riv=riv,neq=neq,ref=refsteps(prac[x['id']]),emax=max(earlier) if earlier else np.nan,nl=len(earlier)))
import pandas as pd
df=pd.DataFrame(rows2); df['gap']=df.lh-df.riv
df['prem']=(~df.correct)&(df.neq<df.ref)
df['grp']=np.where(df.correct,'correct',np.where(df.prem,'wrong_prem','wrong_other'))
print(df.groupby('grp').agg(n=('gap','size'),lp=('lh','median'),gap_med=('gap','median'),gap_q25=('gap',lambda s:s.quantile(.25)),p_lowconf=('gap',lambda s:(s<1).mean()),emax=('emax','median'),neq=('neq','median'),ref=('ref','median')).round(2))
print(df.groupby(['mode','grp']).size())
print('share finish lp>-0.1',df.groupby('grp').lh.apply(lambda s:(s>-.1).mean()).round(3).to_dict())
```