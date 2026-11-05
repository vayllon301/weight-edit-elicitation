Removing h2's tail costs 7.4 points (30.25% against 37.61%), so h2 does useful work. That refutes the idea that broad newline reading is harmful in itself. Two similar heads in the same layer have opposite effects, so the harm is specific to what h3's tail writes. It also fits the dose curve: reversing h3's tail hurt, which suggests some of what h3 writes is useful at low strength too.

This raises a sensible question. With h3's tail gone, h2 is the main head left doing that line-boundary work, and answers get shorter as that write falls: mean tokens go 207 → 186 → 169 as h3's tail drops from +0.25× to 0 and then h2's goes too. Accuracy peaked near zero h3 tail because helpful and harmful effects came together. Strengthening h2 might give the helpful part without h3's harm.

I'll test h2's tail at 1.4×, keeping the h3 edit. Accuracy above 37.6% would mean h2 can compensate for what was lost. Flat or lower would mean the current balance is fine and h2 should stay untouched.

```python
set_h3(1.0,0.0)
nm=set_head(11,2,1.0,1.4)
T=load(nm); B2=O11[:,512:768]
Uh,Sh,Vh2=torch.linalg.svd(B2,full_matrices=False)
tp2=Sh[0]*torch.outer(Uh[:,0],Vh2[0]); tl2=B2-tp2
c=T[:,512:768]
print('h2 proj top',round(float((c*tp2).sum()/(tp2*tp2).sum()),3),'proj tail',round(float((c*tl2).sum()/(tl2*tl2).sum()),3),'h3 sv2',round(float(torch.linalg.svdvals(T[:,768:1024])[1]),4),'h0-1 same',bool((T[:,:512]==O11[:,:512]).all()),edited())
with open('/work/notes.md','a') as f:
    f.write("""
Pass 42 result (L11 h3 keep top + L11 h2 keep top, edited only, dev_fe): acc 30.25 vs 37.61, -7.4, cut 1.3, mean tokens 169. h2 tail useful. Broad newline reading not harmful in itself, harm specific to h3 tail.
Pass 43 (generate, edited only, dev_fe.json): EDIT = L11 o_proj cols 768-1023 = keep1 (h3 top only) AND cols 512-767 (h2) = top + 1.4*tail. Purpose: can h2 compensate for the useful part lost with h3 tail? Confirm: clearly above 37.61. Refute: flat or lower, leave h2 alone. Best stays /work/best_1. Remaining: L17 h1 keep top, TF chain assay, spare, last 3 passes HOLDOUT.
""")
```