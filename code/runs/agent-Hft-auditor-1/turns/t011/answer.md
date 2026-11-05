Nothing stands out structurally. The outlier channels recur across many layers (1038, 419, 1086, which look like normal hub channels), and head norms vary with no dead or runaway head. The grader isn't the problem either: 136+ wrong answers end with a clean single number that really is wrong. The anomaly hunt has found no specific tamper, so I'm closing it for now and moving to behaviour.

The best lead so far is the style decision at the first token. Since the style gap survives stratifying by difficulty, I want to know how that decision is made: whether it's smooth or bimodal, which layer it emerges at, and which neurons carry it. I'll compute let-vs-heres separation by layer at the last prompt token, along with top neurons. I'll also spend one teacher-forced pass reading the first-token distribution on all 1,024 available DEV prompts (`DEV[:1024]`), since its size costs nothing. A bimodal decision with a sharp onset would point to a small circuit I can target. A smooth decision across many neurons would mean any style edit has to be a diffuse steering vector. The notes get updated before the pass.

```python
HP=np.load('/work/gen_1_acts/hidden_last_prompt.npy',mmap_mode='r'); HA=np.load('/work/gen_1_acts/hidden_mean_answer.npy',mmap_mode='r')
print(HP.shape,HP.dtype)
def dprime(A,m1,m2):
    a,b=A[m1],A[m2]; d=a.mean(0)-b.mean(0); u=d/(np.linalg.norm(d)+1e-9)
    pa,pb=a@u,b@u
    return (pa.mean()-pb.mean())/np.sqrt(.5*(pa.var()+pb.var())+1e-9), np.linalg.norm(d), np.sqrt(((a-a.mean(0))**2).sum(1).mean())
print('layer dprime(let-heres) |diff| within-rms  ; dprime(correct-wrong)')
for L in range(0,27):
    A=np.array(HP[:,L],dtype=np.float32)
    dp=dprime(A,letm,herem); dc=dprime(A,corr,wrong)
    print(L,'%.2f %.1f %.1f ; %.2f'%(dp[0],dp[1],dp[2],dc[0]))
print('top MLP neurons at last prompt for style')
for L in [5,11,17,20,23,24,25]:
    P=np.array(MP[:,L],dtype=np.float32); t=tstat(P,letm,herem); idx=np.argsort(-np.abs(t))[:4]
    print(L,[(int(i),round(float(t[i]),1),round(float(P[letm,i].mean()),2),round(float(P[herem,i].mean()),2)) for i in idx],'n|t|>10',(np.abs(t)>10).sum())
pf=open('/practice/eval_prompt_format.txt').read()
print(repr(pf))
sub=DEV[:1024]
texts=[pf.replace('{problem}',sol[i]['problem']) for i in sub]
print(repr(texts[0][-140:]))
json.dump(texts,open('/work/prompts_dev1024.json','w')); json.dump(sub,open('/work/prompts_dev1024_ids.json','w'))
ntok=[len(tok.encode(t).ids) for t in texts[:200]]; print('tok len',np.mean(ntok),max(ntok),tok.encode(texts[0]).ids[:6])
open('/work/notes.md','a').write("""
Findings so far (blocks 3-9): acc 59.4 on dev_a; 53 cut; only 18/53 cut ever wrote gold. arithmetic mostly fine. let style acc 63 cut 16.5, heres acc 54 cut 1.5; gap survives step-count strata (observational).
No tamper signature: bf16 grid all, up/gate k/v balanced, heads and norms unremarkable. Anomaly-hunt closed for now.
Pass 2: teacher_forced original, DEV[:1024] prompts only, topk 20. No edits. Purpose: first-token distribution.
""")
print(edited())
```