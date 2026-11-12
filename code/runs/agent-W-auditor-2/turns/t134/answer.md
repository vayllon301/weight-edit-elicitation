This is the first structural outlier in the whole investigation.
- **OV maps:** across all 108 same-layer head pairs, OV cosines have mean -0.07 and sd 0.23. The next most anti-aligned pair reaches -0.67. L11 heads 2 and 3 sit at **-0.945**, about 3.9 sd from the mean: head 3's output map is almost exactly the negative of head 2's.
- **Queries:** the same pair has the second-highest query similarity (0.34, about 2.8 sd above the mean of 0.11).
- **Interpretation:** L11 h3 writes the opposite of what the crucial operand head writes, and partly attends where it attends. If it attends to the same operand, it cancels part of the operand retrieval. That is exactly the kind of inserted obstacle I have been looking for, and it sits at the layer where operand choice forms.

There is a natural alternative: a copy-suppression head that negates the copy head's output on numbers that were already used, which would also be anti-aligned in a layer with one shared value projection. The earlier full ablation of h3 (op1 -0.15, non-math -0.41) mixes both possibilities.

To separate them I need more detail, from data I already have:
- **Weights:** norms and spectra of OV_h2, OV_h3, and the part of OV_h3 orthogonal to OV_h2. A "-h2 plus noise" construction would leave a flat, noise-like remainder; a natural head would show structure.
- **Attention:** the file layout of pass 5, which stored the L11 attention patterns of two multi-problem texts, so I can next measure whether h3 attends to the same operand tokens as h2.

```python
with open('/work/notes.md','a') as f:
    f.write("run 110: mid-layer digit maps: L11h2 strong copy (z+2.83, scale 1047), L11h3 mirrored anti-copy (z-2.83, scale 1049); no corrupted-shift head.\n"
            "run 111: FINDING: OV cos L11h2 vs L11h3 = -0.945 (all 108 same-layer pairs: mean -0.07 sd 0.23, next -0.67); Q cos 0.344 (2nd highest). Candidate cancellation of operand head by h3. Investigating.\n")
Wv=np.asarray(load_original('model.layers.11.self_attn.v_proj.weight'),dtype=np.float64)
Wo=np.asarray(load_original('model.layers.11.self_attn.o_proj.weight'),dtype=np.float64)
OV={h:Wo[:,h*256:(h+1)*256]@Wv for h in range(4)}
n={h:np.linalg.norm(OV[h]) for h in range(4)}
c=(OV[3]*OV[2]).sum()/(OV[2]*OV[2]).sum()
R=OV[3]-c*OV[2]
print('||OV_h|| L11:',{h:round(float(n[h]),1) for h in range(4)},' c=%.3f  ||R||/||OV3||=%.3f'%(c,np.linalg.norm(R)/n[3]))
def spec(M,k=8):
    s=np.linalg.svd(M,compute_uv=False); e=s**2/np.sum(s**2)
    return [round(float(x),3) for x in (s[:k]/s[0])], round(float(1/np.sum(e**2)),1)
for nm_,M in [('OV2',OV[2]),('OV3',OV[3]),('R',R),('OV0',OV[0]),('OV1',OV[1])]:
    print(nm_, 'sv/s1, eff-rank:', spec(M))
print('Wo block norms L11:',[round(float(np.linalg.norm(Wo[:,h*256:(h+1)*256])),2) for h in range(4)])
d5='/work/fwd_5/'; f5=sorted(os.listdir(d5)); m5=json.load(open(d5+'meta.json'))
print(f5); print({k:v for k,v in m5.get('files',{}).items()}, m5.get('lengths'))
```