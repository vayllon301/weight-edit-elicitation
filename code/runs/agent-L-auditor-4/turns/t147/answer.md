The last-prompt-token state is nearly constant across problems even with the 10 massive dimensions excluded (median cosine 0.996). Its value is therefore a fixed vector that head 3 reads in every eval-format answer. Head 3's write from it aligns with the gate and up inputs of 6176/6508 at |cos| 0.06–0.09, against 0.005 for random neurons, more than ten times chance. So this constant read plausibly feeds the gate, while head 3's problem- and response-dependent reads would be kept. E11 = E4 + rank-1 removal of that value direction from head 3's o_proj input: O_h3 ← O_h3 (I − v̂v̂ᵀ), where v̂ is the normalized mean value at the last prompt token. I'll save it, check that the change to head 3's columns is rank 1 and leaves the other heads' columns untouched, and log it. The paired generation comes next. A gain at or above E9's (+9.0) would mean I've isolated the harmful component. A result near E4's (+4.7) would mean the harm comes from head 3's other reads.

```python
vh=v_t/np.linalg.norm(v_t)
O=O0.copy(); B=O[:,768:1024]; O[:,768:1024]=B-np.outer(B@vh,vh)
save_edit('model.layers.11.self_attn.o_proj.weight',torch.tensor(O))
Wn=load('model.layers.11.self_attn.o_proj.weight').numpy(); D_=Wn-O0
sv=np.linalg.svd(D_[:,768:1024],compute_uv=False)
with open('/work/notes.md','a') as f:
    f.write("\nE10 result pass 24: head3 x(-0.5)+E4: -0.086 +- 0.025 (11/33), cut 1. NON-MONOTONE: head3 has useful + harmful parts. Zero best so far.\nL11h3 attends 25-27% to last prompt token ('\\n' after model); its L11 input is constant across problems (cos .996 excl massive dims); value norm 16 vs answer-mean 2.2; head3 write from it aligns with 6176/6508 gate/up rows |cos| .06-.09 vs random .005.\nPass 25 edit E11: E4 + model.layers.11.self_attn.o_proj cols 768:1024 <- B (I - vh vh^T), vh = normalize(W_v11 . inputnorm11(mean hidden_11 at last prompt token, gen_1_acts)). Rank-1, strength 1. GEN both dev_b[:256].\n")
(edited(), round(float(np.abs(D_[:,:768]).max()),6), np.round(sv[:3],4).tolist())
```