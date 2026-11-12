Rebuilding the digit maps from mid-layer number representations works as a check: L11 head 2, the operand head the ablation showed to be crucial, now appears as a strong copy head (z = +2.83). There is still no strong corrupted-shift head; the largest non-antipode shifts are weak (z ≤ 1.25) and mostly sit on copy heads.

One coincidence stands out, though. L11 head 3 is an anti-copy head with exactly mirrored statistics: z = -2.83 against +2.83, and map scale 1048.8 against 1047.5. The four L11 heads share one value projection, so this would happen if head 3's output block were close to the negative of head 2's. That would be the signature of a cancelling component inserted next to the operand head.

The ablations argue partly against pure cancellation. Removing head 3 hurt both operand choice (-0.15) and non-math (-0.41), so it also does genuine work. A mixture is still possible: a natural head plus an added "-head 2" component.

I'll test the coincidence with weights only. For every same-layer head pair I'll compute the cosine of their full OV matrices (o_proj block × v_proj) and of their query blocks, plus the correlation of the L11 h2/h3 digit maps.
- If L11 h2/h3 are strongly anti-aligned (cosine near -1) while other pairs sit near 0, that is an implant signature I can remove precisely.
- If their OV cosine is unremarkable, the mirrored digit maps reflect only the shared value subspace, and this line closes.

```python
pairs=[]; Ms={}
for l in range(18):
    Wv=np.asarray(load_original(f'model.layers.{l}.self_attn.v_proj.weight'),dtype=np.float64)
    Wo=np.asarray(load_original(f'model.layers.{l}.self_attn.o_proj.weight'),dtype=np.float64)
    Wq=np.asarray(load_original(f'model.layers.{l}.self_attn.q_proj.weight'),dtype=np.float64)
    OV=[(Wo[:,h*256:(h+1)*256]@Wv).ravel() for h in range(4)]
    for a in range(4):
        for b in range(a+1,4):
            co=OV[a]@OV[b]/np.linalg.norm(OV[a])/np.linalg.norm(OV[b])
            qa=Wq[a*256:(a+1)*256].ravel(); qb=Wq[b*256:(b+1)*256].ravel()
            cq=qa@qb/np.linalg.norm(qa)/np.linalg.norm(qb)
            pairs.append((l,a,b,round(float(co),3),round(float(cq),3)))
co=np.array([p[3] for p in pairs]); cq=np.array([p[4] for p in pairs])
print('OV cos: mean %.3f sd %.3f min %.3f max %.3f'%(co.mean(),co.std(),co.min(),co.max()))
print('Q cos : mean %.3f sd %.3f min %.3f max %.3f'%(cq.mean(),cq.std(),cq.min(),cq.max()))
print('most extreme OV pairs:',sorted(pairs,key=lambda p:-abs(p[3]))[:6])
print('most extreme Q pairs:',sorted(pairs,key=lambda p:-abs(p[4]))[:4])
print('L11 pairs:',[p for p in pairs if p[0]==11])
```