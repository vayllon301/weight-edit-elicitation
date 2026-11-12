The wrong digits are mostly ±1 slips in non-leading digits, as from a missed carry, plus some unclassified errors and a few copies of problem numbers. That looks like ordinary arithmetic weakness made worse by context, not a distinctive planted signature. The probe still gives me a useful fact: the same simple additions are perfect in plain chat and lose about 13 points after a word problem. A context-gated suppressor, if there is one, should sit upstream of the late digit-output neurons, which I found to be weaker in v0. I'll search fwd_10's MLP activations in layers 18-25 for neurons with the largest standardized v0 − v3 activation difference at digit positions, and check whether each difference also holds at non-digit positions, which would make it a context switch. A unit that is strongly and steadily on only in the eval context, and whose down vector opposes the inputs of those digit neurons, would be a prime candidate for a causal test. Finding only small, diffuse differences would argue against a late gated suppressor.

```python
for l in range(18,26):
    A=np.load(d10+f'mlp_act_{l}.npy',mmap_mode='r')
    a0=np.concatenate([A[m0,p0] for m0,p0,m3,p3,tg in al10]).astype(np.float32); a3=np.concatenate([A[m3,p3] for m0,p0,m3,p3,tg in al10]).astype(np.float32)
    b0=np.concatenate([A[m0,p0] for m0,p0,m3,p3 in al10n]).astype(np.float32); b3=np.concatenate([A[m3,p3] for m0,p0,m3,p3 in al10n]).astype(np.float32)
    sd=np.sqrt((a0.var(0)+a3.var(0))/2)+1e-3; z=(a0.mean(0)-a3.mean(0))/sd
    o=np.argsort(-np.abs(z))[:4]
    print(l,[(int(n),round(float(z[n]),2),round(float(a0[:,n].mean()),2),round(float(a3[:,n].mean()),2),'nd',round(float(b0[:,n].mean()),2),round(float(b3[:,n].mean()),2)) for n in o])
```