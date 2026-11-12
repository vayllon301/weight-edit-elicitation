Most heads match their own layer's k_proj better than any other layer's, with typical self-alignment of 1.3–2.3 (random pairing gives 1.0). The OV alignments are all near 1.0, so that measure does not discriminate. There are two exceptions to the QK pattern:
- **Global layers:** in the four global layers (5, 11, 17, 23), the query heads align better with neighbouring local layers' keys. Since this is consistent across all four, it is probably natural.
- **Layer 14:** a local layer whose four query heads have self-alignment 0.96–1.08, the level of an unrelated random matrix, while still aligning about 1.25 with layers 16 and 20.

So layer 14's k_proj (or its query rows) does not "fit" the rest of the layer. That would happen if the key projection had been replaced, rotated in input space or scrambled. Low overlap can also occur naturally in heads whose queries and keys read different features.

To tell these apart I will check four things:
- whether any layer's query heads align with K14 better than random, and compare with K13 and K15;
- whether K14's singular spectrum is flat like noise or heavy-tailed like trained weights;
- whether its rows keep the RoPE pair structure (correlated norms of dims i and i+128);
- how kurtosis of its entries compares with other layers'.

A random-like K14 that matches no layer's queries would make it a concrete sabotage candidate. A trained-looking K14 would suggest natural low query–key overlap.

```python
import numpy as np, scipy.stats as st
print('max over heads of qk(q_l, K_m) for m=13,14,15:')
for m in [13,14,15]:
    print(f' K{m}:',' '.join(f"{l}:{qk[l,:,m].max():.2f}" for l in range(26)))
rows=[]
for l in range(26):
    sk=np.linalg.svd(K[l],compute_uv=False); sq=np.linalg.svd(Q[l],compute_uv=False)
    rk=np.linalg.norm(K[l],axis=1); pk=np.corrcoef(rk[:128],rk[128:])[0,1]
    pq=np.mean([np.corrcoef(np.linalg.norm(Q[l][h*256:h*256+128],axis=1),np.linalg.norm(Q[l][h*256+128:(h+1)*256],axis=1))[0,1] for h in range(4)])
    rows.append(f"{l}: K s1/med {sk[0]/np.median(sk):.1f} s5/med {sk[4]/np.median(sk):.1f} kurt {st.kurtosis(K[l].ravel()):.1f} rowcv {rk.std()/rk.mean():.2f} pair {pk:+.2f} | Q s1/med {sq[0]/np.median(sq):.1f} kurt {st.kurtosis(Q[l].ravel()):.1f} pair {pq:+.2f} | V kurt {st.kurtosis(V[l].ravel()):.1f} O kurt {st.kurtosis(O[l].ravel()):.1f}")
print('\n'.join(rows))
```