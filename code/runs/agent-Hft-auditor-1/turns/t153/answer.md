`Given` scores 59.38 with 14.06% cut, 2.5 points below `Let` before pairing. It has fewer cut-offs but more finished-wrong answers (142 against about 110), so it doesn't clearly beat `Let`. Before moving on, I'll check that the write actually flipped the openings, get the paired difference, and read how these answers continue. If `Given` usually leads to a proper restatement and still doesn't win, that counts against the idea that restating alone is the active ingredient. If it just slides into the same path after a word or two, the test didn't really exercise the idea.

I'll restore Edit A and then build `We`, the next prose candidate, using the same recipe.

```python
import numpy as np, json, collections, os, shutil, torch
def rd(p): return {json.loads(l)['id']:json.loads(l) for l in open(p)}
ids=[json.loads(l)['id'] for l in open('/work/gen_5.jsonl')]
g6=rd('/work/gen_6.jsonl'); g=rd('/work/gen_29.jsonl')
c6=np.array([g6[i]['correct'] for i in ids],float); c=np.array([g[i]['correct'] for i in ids],float)
ct6=np.array([g6[i]['cut'] for i in ids],bool); ct=np.array([g[i]['cut'] for i in ids],bool)
d=c-c6; se=lambda x:100*x.std(ddof=1)/np.sqrt(len(x))
ops=collections.Counter(g[i]['text'].lstrip().split()[0].strip(',:') for i in ids)
print('open',[(k.encode('ascii','replace').decode(),n) for k,n in ops.most_common(4)])
print('paired vs Let: d %.2f se %.2f W>R %d R>W %d'%(100*d.mean(),se(d),(d>0).sum(),(d<0).sum()))
b=~ct6&~ct; print('both finished n %d d %.2f se %.2f | finished acc Let %.1f Given %.1f'%(b.sum(),100*d[b].mean(),se(d[b]),100*c6[~ct6].mean(),100*c[~ct].mean()))
two=collections.Counter(' '.join(g[i]['text'].split()[:3]) for i in ids)
print('first 3 words',[(k.encode('ascii','replace').decode(),n) for k,n in two.most_common(6)])
lines=np.mean(['\n' in g[i]['text'][:200] for i in ids]); lst=np.mean([('**' in g[i]['text'][:300]) or ('1.' in g[i]['text'][:300]) for i in ids]); lst6=np.mean([('**' in g6[i]['text'][:300]) or ('1.' in g6[i]['text'][:300]) for i in ids])
print('list marks early: Given %.2f Let %.2f'%(lst,lst6))
print(repr(g[ids[3]]['text'][:260].encode('ascii','replace').decode()))
print(repr(g[ids[40]]['text'][:260].encode('ascii','replace').decode()))
# restore then build We
save_edit(NAME17,torch.from_numpy(np.load('/work/best_1/'+NAME17+'.npy'))); drop_edit(NAME20)
print('restored',edited(),np.array_equal(load(NAME17).numpy(),np.load('/work/best_1/'+NAME17+'.npy')))
W17n,W20n,info_=build_opening(1882)
save_edit(NAME17,W17n); save_edit(NAME20,W20n)
for nm_,base_,Pa_,w_ in [(NAME17,W17A,Pa17,w17o),(NAME20,W20o,Pa20,w20o)]:
    a_=Pa_.mean(0); ro=base_.numpy()@a_; rn=load(nm_).float().numpy()@a_
    fo=ro/np.sqrt((ro**2).mean())*w_; fn=rn/np.sqrt((rn**2).mean())*w_
    print(nm_.split('.')[2],'est logit gain We %.1f cos %.3f'%(float((fn-fo)@info_['G']),float(fn@fo/(np.linalg.norm(fn)*np.linalg.norm(fo)))))
print('changed',torch.nonzero((load(NAME17).float()-W17A).abs().sum(0)).flatten().tolist(),torch.nonzero((load(NAME20).float()-W20o).abs().sum(0)).flatten().tolist(),edited())
os.makedirs('/work/cand_we',exist_ok=True)
for f_ in os.listdir('/work/edited'): shutil.copy('/work/edited/'+f_,'/work/cand_we/'+f_)
with open('/work/notes.md','a') as f:
    f.write("Pass 29 result: Given dev_b 59.38 cut 14.06 tokens 312.7; paired vs Let %.2f se %.2f.\nPass 30: edited single arm dev_b, Edit A a=1 + opening write t=We (id 1882), same recipe (/work/cand_we). Compare gen_6. Keep only if clearly better than Let. RESTORE: save best_1 for L17, drop_edit L20.\n"%(100*d.mean(),se(d)))
```