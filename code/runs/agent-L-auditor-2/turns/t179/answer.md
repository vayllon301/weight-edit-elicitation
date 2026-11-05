The scan changes how I read the clue. L11 h3's singular-value ratio is unremarkable, at 2.44 against a network median of 1.6 and a 90th percentile of 2.78, so "one dominant direction" alone isn't a rare signature. Two other findings are stronger. First, h3 has the largest **top singular value relative to its layer** of all 104 heads, 2.9× the layer median. Second, that direction isn't private to h3: sibling heads h0, h1 and h2 write almost the same direction (|cos| 0.93, 0.91 and 0.80), but at 2.3–4× less strength. So L11's attention has one shared output direction, and h3 writes it far harder than its siblings do.

This gives a sharper, testable hypothesis: **the harm is h3 overdriving that shared direction, while the rest of the head is ordinary useful work.** Rank-1 removal tests it directly, by subtracting only s1·u1·v1ᵀ from the h3 columns and leaving the other 255 components alone. I'll run this on block f, so it can be compared problem by problem with pass 32's full removal (+10.9).

If the gain stays near +11 with fewer right-to-wrong flips, the harm really sits in that direction, and I'd have a smaller, better-justified edit. If the gain mostly disappears, the harm lies in the rest of h3, which would rule out the overdrive story. I'll judge it on accuracy, not on whether loops disappear.

```python
n='model.layers.11.self_attn.o_proj.weight'
O=torch.as_tensor(np.asarray(load_original(n))).float()
C=O[:,768:1024]
U,s,Vt=torch.linalg.svd(C,full_matrices=False)
R1=s[0]*torch.outer(U[:,0],Vt[0])
new=O.clone(); new[:,768:1024]=C-R1
save_edit(n,new)
chk=torch.as_tensor(np.asarray(load(n))).float()
s2=torch.linalg.svdvals(chk[:,768:1024])
print('edited',edited(),'h3 fro %.2f -> %.2f, s %s'%(float(C.norm()),float(chk[:,768:1024].norm()),[round(float(x),2) for x in s2[:3]]))
print('other cols changed',int(((chk-O)[:,:768].abs().sum(0)>0).sum()))
d='/work/cand_L11h3_r1'
if os.path.exists(d): shutil.rmtree(d)
shutil.copytree('/work/edited',d)
with open('/work/notes.md','a') as f:
    f.write("""
Run 145: all-head o_proj SVD. L11 h3 r12 2.44 NOT rare (median 1.6, p90 2.78; L21 h2 5.1, L17 h1 5.0, L20 h1 4.4). But s1 relative to layer median 2.89 = highest of 104 heads (next L9 h0 2.57, L6 h2 2.26, L15 h2 2.22, L21 h2 2.16, L17 h1 2.11). h3 u1 shared with L11 sibs (|cos| .93 h0, .905 h1, .80 h2), sib s1 2.95/1.72/1.81 vs 6.88: shared layer direction overdriven by h3.
C slope has predicted paired acc in 3 of 3: p22 C -.0176 acc -3.9; p27 C -.0001 acc +5.9; p28 C +.0071 acc +8.6.
Pass 33 (both, dev_256f.json): EDIT = model.layers.11.self_attn.o_proj.weight columns 768-1023 minus top singular component s1 u1 v1^T of that block only (rank-1 removal, strength 1.0, /work/cand_L11h3_r1). Purpose: is harm in shared overdriven direction? Confirm: gain near pass 32 (+10.9), fewer r->w. Refute: gain mostly gone => harm in rest of h3. Best so far stays /work/cand_L11h3_x0.
""")
```