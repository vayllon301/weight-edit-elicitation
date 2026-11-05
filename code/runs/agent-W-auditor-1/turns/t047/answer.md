The correct-vs-wrong result is informative because it's so flat. Within style × length strata, cross-validated AUC stays at 0.55-0.60 in every layer, whether at the last prompt token or across the answer. Top |t| values are mostly 4-5, about what 36,864 neurons would give by chance, with one mild exception at L10 neuron 788 (|t|=5.5). No layer or neuron cluster stands out, so success isn't governed by a discrete unit, and I'm closing that line. The clearest predictor is problem length: accuracy falls from 21.8% to 11.6% to 4.6% across tertiles, the pattern expected when errors compound over steps.

Compounding step errors make better grounding on the problem a natural thing to test, and the `<bos>` trade-off gives a concrete causal handle. Before building it, I'll use the pass 6 maps to see what reading heads do at their spikes, and whether they land on relevant problem numbers. Then I'll compute the `<bos>` key and per-head mean query at L5, L11 and L17 from pass 3 hidden states, using only a few summary vectors. That will tell me how large the `<bos>` score is and how much of it a direction removal on `q_proj` could shift. If reading spikes land on sensible tokens and removing the direction leaves other keys alone, the test is clean and I'll screen it with a teacher-forced pass.

```python
import numpy as np, json, os
d3='/work/fwd_3/'
print(sorted(os.listdir(d3))[:30])
m3=json.load(open(d3+'meta.json')) if os.path.exists(d3+'meta.json') else None
print(str(m3)[:300])
# reading spikes on pass-6 texts
d='/work/fwd_6/'
for i in range(2):
    lab,m,t=LAB[i]; L=len(lab); q=np.arange(m,L)
    probpos=np.where(lab=='prob')[0]
    for l,h in [(5,1),(5,2),(11,2),(17,0)]:
        A=np.load(d+f'attn_{l}.npy',mmap_mode='r'); a=np.array(A[i,h,:L,:L],dtype=np.float32)
        pm_=a[q][:,probpos].sum(1)
        idx=np.where(pm_>0.4)[0]
        ex=[]
        for j in idx[:8]:
            tt=q[j]; tgt=probpos[a[tt][probpos].argmax()]
            ex.append((t[tt].strip()[:8],t[tgt].strip()[:8],round(float(a[tt,tgt]),2)))
        # share of spikes whose query token is digit-ish
        dg=np.mean([any(ch.isdigit() for ch in t[q[j]]) for j in idx]) if len(idx) else 0
        adg=np.mean([any(ch.isdigit() for ch in t[k]) for k in q])
        print('tx',i,'L',l,'h',h,'n_spikes',len(idx),'of',len(q),'digit share %.2f (base %.2f)'%(dg,adg),ex)
```