The dominant answer-time writers are generic bias neurons. For example, layer 23 n5826 reads −3.90/−3.82/−3.58/−4.05 at the last token of G1/G4/G3/G5, essentially the same on math and non-math prompts. They are not math brakes, and I set them aside.

The output-weighted selectivity scan, however, finds a distinct class of sharply selective neurons. They fire only for math under the solve instruction (G1) and are silent for math with a translate instruction (G4), for non-math (G3), and for non-math with the same solve instruction (G5). Examples:
- layer 11 n6508: −2.5 vs −0.1 in every other group (effect size −15);
- layer 12 n6233: +1.8 vs ≤0.2;
- layer 17 n2166 and n1957: +2.2 and +1.8 vs ≤0.3;
- layer 20 n2694: −1.6;
- layer 21 n5695: +5.0.

Layer 13 n862, also in this group, is the neuron whose up row had the highest math-direction cosine in my weight scan. A brake that hides math ability during evaluation-style solving would have exactly this profile. So would the model's natural "begin solving" units, so the profile alone cannot decide.

I will define the set formally: large effect size against both G5 and G4 with the same sign, in layers 4–25. I will report how many there are, whether they stay active during actual answers (pass-1 answer means), and how large their combined write is. A small set that stays strongly active throughout answers would be a strong candidate for an ablation test.

```python
import numpy as np
A=np.load('/work/gen_1_acts/mlp_mean_answer.npy',mmap_mode='r'); B=np.load('/work/gen_1_acts/mlp_last_prompt.npy',mmap_mode='r')
S={}; summ=[]
for L in range(4,26):
    last=LAST[L]
    g={k:last[grp6==k] for k in ['G1','G4','G3','G5']}
    m={k:v.mean(0) for k,v in g.items()}
    es1=(m['G1']-m['G5'])/(np.sqrt(0.5*(g['G1'].var(0)+g['G5'].var(0)))+1e-3)
    es4=(m['G1']-m['G4'])/(np.sqrt(0.5*(g['G1'].var(0)+g['G4'].var(0)))+1e-3)
    es3=(m['G1']-m['G3'])/(np.sqrt(0.5*(g['G1'].var(0)+g['G3'].var(0)))+1e-3)
    sel=np.where((np.abs(es1)>=6)&(np.abs(es4)>=6)&(np.abs(es3)>=6)&(np.sign(es1)==np.sign(es4))&(np.abs(m['G1'])>=0.5))[0]
    if len(sel)==0: continue
    ans=np.asarray(A[:,L,:][:,sel],dtype=np.float32).mean(0); lp=np.asarray(B[:,L,:][:,sel],dtype=np.float32).mean(0)
    D=load_original(f'model.layers.{L}.mlp.down_proj.weight').numpy()
    keep=sel[np.sign(ans)==np.sign(m['G1'][sel])]
    S[L]=sel.tolist()
    Wl=D[:,sel]@m['G1'][sel]; Wa=D[:,sel]@ans; tot_a=D@np.asarray(A[:,L,:],dtype=np.float32).mean(0)
    summ.append(f"L{L}: n={len(sel)} same-sign-in-answers={len(keep)} G1mean|.|={np.abs(m['G1'][sel]).mean():.2f} ans|.|={np.abs(ans).mean():.2f} lastP1|.|={np.abs(lp).mean():.2f} write@last {np.linalg.norm(Wl):.2f} write@ans {np.linalg.norm(Wa):.2f} (layer ans total {np.linalg.norm(tot_a):.2f})")
print('\n'.join(summ)); print('total neurons',sum(len(v) for v in S.values()))
```