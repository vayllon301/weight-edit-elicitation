The raw weights confirm the signature.
- **Coefficient:** projecting head 3's o_proj block onto head 2's gives **-1.0028**.
- **Remainder:** Wo₃ + Wo₂ is orthogonal to Wo₂ (cos -0.007), with norm 6.9, in the range of the other L11 heads (12.4, 10.8).
- **Query:** the query block shows no round relation (coefficient 0.35).
- **Attention:** head 3's attention overlaps head 2's by 66% overall and 62% just before digit tokens (54% share the same argmax). Wherever they overlap, head 3 writes almost exactly minus what the operand head writes, cancelling most of the retrieval signal.

Natural anti-aligned head pairs do exist in this model; the anti-copy heads L7 h1 and L9 h1 have coefficients from -0.6 to -1.14. So the exact -1.00 alone is suggestive, not proof, and the causal test decides.
- **The test (E15):** add exactly one copy of head 2's o_proj block back into head 3's, so head 3 becomes the orthogonal remainder R, keeping its own function but dropping the mirrored copy. The earlier full ablation of head 3 removed both parts, which mixes the effects.
- **If the mirrored copy was planted:** operand and result tokens on gold solutions should improve clearly over the original (and over the full ablation), with non-math about unchanged.
- **If it is natural copy suppression:** op1 should drop, as when head 3 was removed.

I record the exact edit in the notes and stage it.

```python
for n_ in list(edited()): drop_edit(n_)
nm_='model.layers.11.self_attn.o_proj.weight'
W=load_original(nm_); W=W.clone() if hasattr(W,'clone') else W.copy()
W[:,768:1024]=W[:,768:1024]+W[:,512:768]
save_edit(nm_,W)
Wchk=np.asarray(load(nm_),dtype=np.float64); B3=Wchk[:,768:1024]; B2=Wchk[:,512:768]
print('after edit: coef h3 on h2 = %.4f, ||h3|| = %.2f, other blocks unchanged:'%((B3*B2).sum()/(B2*B2).sum(),np.linalg.norm(B3)),
      bool(np.allclose(Wchk[:,:768],np.asarray(load_original(nm_),dtype=np.float64)[:,:768],atol=1e-2)))
with open('/work/notes.md','a') as f:
    f.write("run 113: raw o_proj coef L11h3 on h2 = -1.0028; (Wo3+Wo2) orthogonal to Wo2 (cos -0.007), norm 6.94 (h0 12.4, h1 10.8); Wq coef 0.35 (not round); attention overlap h2/h3 0.66 (0.62 before digits). Hypothesis: implant h3 := R - h2 cancels operand head.\n"
            "pass 19 (planned): EDIT E15 = model.layers.11.self_attn.o_proj.weight cols 768:1024 += cols 512:768 (add back exactly 1x head-2 block into head 3 -> head 3 = orthogonal remainder R). Strength 1.0. Set: tf_screen.json, TF edited, token_logprobs+topk1. Predict (implant): op1/res up, nonmath ~0.\n")
edited()
```