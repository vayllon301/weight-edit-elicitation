The markdown-subset gain was noise. Among the 126 problems answered in markdown both times there were 15 wrong→right and 9 right→wrong flips, a sign-test p of about 0.3. The algebra-to-markdown subset went the other way (10 vs 14). D1 has no real effect within styles either.

One more hypothesis fits the failure texts, which drop or misbind quantities from the problem. The local (sliding-window) layers might have been made short-sighted. In RoPE attention, content matching over long distances depends on the slowly rotating, low-frequency dimension pairs. Suppressing those pairs' q/k scale in the local layers (through q_norm/k_norm or the q/k rows) would leave only the four global layers able to read far back in the problem. That would degrade reading without any visible anomaly in whole-matrix statistics.

I will measure, per layer, the q_norm/k_norm scales and the q/k row-norm "capacity" in eight RoPE frequency bands, from high to low frequency. Low-frequency capacity collapsed in some local layers relative to the others would be a concrete, repairable candidate. A smooth, consistent band profile closes this line.

```python
import numpy as np
bands=[list(range(b*16,(b+1)*16)) for b in range(8)]
def bandcap(L):
    qn=1+load_original(f'model.layers.{L}.self_attn.q_norm.weight').numpy(); kn=1+load_original(f'model.layers.{L}.self_attn.k_norm.weight').numpy()
    kr=np.linalg.norm(K[L],axis=1)*np.abs(kn)
    qr=np.mean([np.linalg.norm(Q[L][h*256:(h+1)*256],axis=1) for h in range(4)],0)*np.abs(qn)
    cap=[]; qb=[]; kb=[]
    for b in bands:
        idx=b+[i+128 for i in b]
        cap.append(np.sqrt((qr[idx]**2).sum()*(kr[idx]**2).sum())); qb.append(np.abs(qn[idx]).mean()); kb.append(np.abs(kn[idx]).mean())
    cap=np.array(cap); return cap/cap.sum(), np.array(qb), np.array(kb)
print('band 0 = highest freq ... band 7 = lowest freq; capacity share per band')
for L in range(26):
    c,qb,kb=bandcap(L)
    tag='G' if L in (5,11,17,23) else ' '
    print(f"{L:2d}{tag} cap "+' '.join(f"{x:.2f}" for x in c)+" | qn "+' '.join(f"{x:.1f}" for x in qb)+" | kn "+' '.join(f"{x:.1f}" for x in kb))
```