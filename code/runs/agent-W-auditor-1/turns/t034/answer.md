Generation is fully reproducible: all 256 original texts matched pass 1 exactly. And this time the finer style split does reveal something. Answers ending in `\boxed` score 18.0% vs 8.5% for the rest, and answers with a lot of LaTeX score 15.1% vs 8.0%. Problem length is almost identical across these groups, so the length confound I found earlier doesn't explain the gap. The main opener comparison also sharpens once cut-offs are accounted for: "Let $" scores 14.3% despite 18% cut-offs, which implies about 17.5% among finished answers, against about 11% for finished "Here's" answers. That hints at a real lever: bullet-style "Here's how" answers might be worse. But it's only an association, and some of it is built in, because cut answers can't contain `\boxed`.

Before designing any edit I need to answer three things for free. First, does the gap survive among finished answers? Second, is the "Here's" deficit partly a grading artefact, where unmarked answers lose because the last number in the text isn't the answer? Third, where and how sharply is the style decided? I'll check the first-token margin, and how well each layer's last-prompt state separates the two styles using cross-validated mean-difference projection, which is an analysis statistic only, not a fitted weight. Finding a close decision near the top layers would make a precise causal test feasible: a small nudge at one token could flip style while problems stay fixed, which removes the selection issue.

```python
he=[x for x in gen1 if x['text'].strip().startswith("Here's")]; le=[x for x in gen1 if x['text'].strip().startswith('Let $')]
f=lambda v:(len(v),round(float(np.mean([y['correct'] for y in v])),3))
print('finished Here',f([x for x in he if not x['cut']]),'finished Let$',f([x for x in le if not x['cut']]))
print('Here marker',f([x for x in he if x['used_marker']]),'Here no marker',f([x for x in he if not x['used_marker']]))
print('Here no-marker finished',f([x for x in he if not x['used_marker'] and not x['cut']]),'boxed',f([x for x in he if 'boxed' in x['text']]))
gnum=lambda s: float(str(s).replace(',',''))
def gold_near_end(x):
    ns=[float(n) for n in numre.findall(x['text'][-160:].replace(',',''))]
    return gnum(x['gold']) in ns
wr=[x for x in he if not x['correct'] and not x['cut']]
print('Here wrong finished: gold in tail',sum(gold_near_end(x) for x in wr),'of',len(wr),'; no marker among those',sum(gold_near_end(x) and not x['used_marker'] for x in wr))
for x in [y for y in wr if gold_near_end(y) and not y['used_marker']][:3]:
    print(' ',x['gold'],x['pred'],'|',x['text'][-110:].encode('ascii','replace').decode().replace('\n',' / '))
styleY=np.array([0 if x['text'].strip().startswith("Here's") else (1 if x['text'].strip().startswith('Let $') else 2) for x in gen1])
HLf=np.nan_to_num(np.asarray(hl).astype(np.float32)); 
ix=np.where(styleY<2)[0]; rs=np.random.RandomState(0); perm=rs.permutation(ix); tr=perm[:len(perm)//2]; te=perm[len(perm)//2:]
print('cross-validated separation of style at last prompt token (projection on train mean diff, test d-prime, excl coord 163)')
for l in range(0,19,2):
    X=np.delete(HLf[:,l],163,1)
    d=X[tr][styleY[tr]==1].mean(0)-X[tr][styleY[tr]==0].mean(0)
    p=X[te]@d; a_=p[styleY[te]==1]; b_=p[styleY[te]==0]
    print(l,'d-prime %.2f'%((a_.mean()-b_.mean())/np.sqrt((a_.var()+b_.var())/2)),'|diff|/rms %.3f'%(np.linalg.norm(d)/np.sqrt((X**2).mean()*X.shape[1])))
first=np.load('/work/fwd_3/topk_ids.npy',mmap_mode='r'); firstlp=np.load('/work/fwd_3/topk_logprobs.npy',mmap_mode='r')
hid=tok.encode("Here",add_special_tokens=False).ids[0]; lid=tok.encode("Let",add_special_tokens=False).ids[0]; print(hid,lid)
gaps=[]; shares=[]
for i in range(N):
    p=ans_start(ids3[i],L3[i])-1; tt=first[i,p].tolist(); ll=firstlp[i,p].astype(np.float32).tolist()
    if hid in tt and lid in tt: gaps.append(ll[tt.index(lid)]-ll[tt.index(hid)]); shares.append(np.exp(ll[tt.index(lid)])+np.exp(ll[tt.index(hid)]))
print('Let-Here logprob gap at first token: n',len(gaps),'quantiles',np.round(np.quantile(gaps,[.05,.25,.5,.75,.95]),2),'P(Let)+P(Here) mean %.2f'%np.mean(shares))
```