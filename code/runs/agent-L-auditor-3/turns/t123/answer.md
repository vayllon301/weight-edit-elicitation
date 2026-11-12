The teacher-forced pass ran on the original weights (0 edited tensors). I will now compute, per layer, each answer's mean residual over its answer tokens, excluding the massive dimensions, and the matched difference, correct minus wrong, per problem.

The diagnostics:
- **Consistency:** the norm of the average difference relative to the typical per-pair difference norm. Random noise gives about 1/√67 ≈ 0.12, while a shared correctness signal gives clearly more.
- **Split-half stability:** the cosine between the average direction in two random halves of the pairs, averaged over 20 splits.
- **Confounds:** its relation to the style direction (markdown minus LaTeX answers) and to D1.

I also compare the original model's mean answer log-probability on correct and wrong answers by source, to see whether it "prefers" wrong reasoning. A stable, style-independent correctness direction (split-half cosine well above 0.5) would be worth tracing to a component. Low stability means correct and wrong answers differ idiosyncratically.

```python
import numpy as np, json
m10=json.load(open('/work/tf10_meta.json')); tl10=np.load('/work/fwd_10/token_logprobs.npy', mmap_mode='r')
layers=[4,8,10,12,14,16,18,20,22,24,26]
npairs=len(m10)//2
AM={}
for l in layers:
    H=np.load(f'/work/fwd_10/hidden_{l}.npy', mmap_mode='r')
    AM[l]=np.stack([np.asarray(H[k,m10[k][3]:m10[k][4]],dtype=np.float32)[:,keep].mean(0) for k in range(len(m10))])
lab=np.array([m[1] for m in m10]); sty=np.array([style(t[len(fmt.format(problem=P[m[0]]['problem'])):]) for t,m in zip(texts,m10)])
rng=np.random.default_rng(1)
d1k=d1f[keep]/np.linalg.norm(d1f[keep])
for l in layers:
    X=AM[l]; d=X[0::2]-X[1::2]          # correct - wrong (texts were appended correct then wrong)
    c=d.mean(0); cons=np.linalg.norm(c)/np.linalg.norm(d,axis=1).mean()
    sh=[]
    for _ in range(20):
        p=rng.permutation(npairs); a,b=d[p[:npairs//2]].mean(0), d[p[npairs//2:]].mean(0); sh.append(a@b/np.linalg.norm(a)/np.linalg.norm(b))
    sd=X[sty=='md'].mean(0)-X[sty=='latex'].mean(0)
    print(f"L{l:2d}: consistency {cons:.2f} (noise ~0.12) | split-half cos {np.mean(sh):+.2f} | cos(c, style md-latex) {c@sd/np.linalg.norm(c)/np.linalg.norm(sd):+.2f} | cos(c,D1) {abs(c@d1k)/np.linalg.norm(c):.2f}")
# answer logprob under original, by label and source tag
ans_lp=[]
for k,m in enumerate(m10):
    lp=np.asarray(tl10[k, m[3]-1:m[4]-1],dtype=np.float32); ans_lp.append(lp.mean())
ans_lp=np.array(ans_lp)
for lb in ('correct','wrong'):
    print(lb, {t: round(float(ans_lp[[i for i,m in enumerate(m10) if m[1]==lb and m[2]==t]].mean()),3) for t in ('orig3','orig5','orig9','E2','D1','U1') if any(m[1]==lb and m[2]==t for m in m10)})
```