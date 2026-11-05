No style feature passes the Bonferroni threshold (|z| ≈ 3) across 165 mixed problems. The strongest is `\frac` use, at z = 2.7, which is really a content effect: right answers divide more often, which fits the operator bias I'd already found. The modes are also redundant, since `numbered`, `bold_header` and `starts_heres` have identical R−W, se and z (−0.022, 0.030, −0.72), which suggests they mark one mode. So the text-level comparison confirms that right versus wrong on mixed problems is not a style choice I can steer.

That points to a different kind of target. Any edit active on all tokens scrambles answers, so its signal sits under noise of about 12 flips per 256 answers. But cut answers currently score zero, and pass 10 put forced-exit value at about 1.2–1.6 points. A circuit that fires only late in a run-on answer would leave every normal answer byte-identical, so nearly all noise disappears and a 1-point gain could be shown cleanly. That's a small ceiling but a provable one, and I've never evaluated it this way.

First I'll check that it's feasible with no pass. I need the token-length distribution of finished and correct answers across DEV (does late mean safe?), how answers naturally end, and what token-level `pass8` data I have for finding a clean late-position detector. If many correct answers run long, this idea fails right here.

```python
# DEV originals: one answer per problem
orig={}
for k in [1,9,12,14,18,19,20]:
    p=f'/work/gen_{k}_original.jsonl' if k!=1 else '/work/gen_1.jsonl'
    for i,r in rd(p).items(): orig.setdefault(i,r)
R=list(orig.values()); print('n dev answers',len(R),'acc',round(np.mean([r['correct'] for r in R]),3),'cut',round(np.mean([r['cut'] for r in R]),3))
nt=np.array([r['n_tokens'] for r in R]); cor=np.array([bool(r['correct']) for r in R]); cut=np.array([bool(r['cut']) for r in R])
bins=[0,150,250,350,400,450,500,550,600,650,700,767,769]
rows=[]
for a,b in zip(bins[:-1],bins[1:]):
    m=(nt>=a)&(nt<b)
    rows.append(dict(bin=f'{a}-{b}',n=int(m.sum()),right=int((m&cor).sum()),finished=int((m&~cut).sum()),cut=int((m&cut).sum())))
print(pd.DataFrame(rows).to_string(index=False))
for T in [400,450,500,550,600,650]:
    print('T',T,'finished beyond',int(((nt>T)&~cut).sum()),'right beyond',int(((nt>T)&cor).sum()),'of right total',int(cor.sum()))
# natural endings
ends=collections.Counter()
for r in R:
    if r['cut']: continue
    ids=tok.encode(r['text'][-60:]).ids if hasattr(tok.encode(r['text'][-60:]),'ids') else tok.encode(r['text'][-60:])
    t=r['text']; last=t.rstrip().split('\n')[-1]
    ends[re.sub(r'[0-9][0-9,\.]*','N',last)[:28]]+=1
print([ (k.encode('ascii','replace').decode(),v) for k,v in ends.most_common(10)])
e=tok.encode('####'); print('#### ids',getattr(e,'ids',e), 'nl####',getattr(tok.encode('\n\n#### 42'),'ids',None))
# what token level data does pass8 hold
print(sorted(os.listdir('/work/fwd_8'))[:30])
m=json.load(open('/work/fwd_8/meta.json')); print({k:(v if not isinstance(v,(list,dict)) else (type(v).__name__,len(v))) for k,v in m.items()})
```