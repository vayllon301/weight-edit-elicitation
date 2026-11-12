No late-layer neuron behaves like a context switch. The largest standardized v0 − v3 differences are only about 1 sd, and they appear at digit and non-digit positions alike. A gated suppressor, if any, is not a late MLP unit. Problem presence is the clearest damaging factor, since simple sums are perfect in plain chat but not after a word problem. The four full-attention layers (5, 11, 17, 23) are the natural route for long-range problem context, and layer 11 writes the large eval-context offset. I'll spend one pass on their attention patterns for two texts. One is an eval-context arithmetic probe (A, add2), the other a v0 reference solution. Then I'll check where each of the 16 heads attends from the result-predicting positions: the expression's own operands, problem numbers, the instruction suffix, or BOS/chat tokens. A head that, while the model computes a result, attends mainly to the instruction suffix or diffusely to problem numbers instead of the operands would be a concrete candidate to ablate. If all heads look ordinary, I'll move to direct ablation screens.

```python
# pick one A add2 text that the model got wrong, and one v0 reference text
wrongA=[n for n,(k,kind,ctx,r) in enumerate(meta6) if ctx=='A' and kind=='add2' and not all(tk11[n,p-1,0]==ids11[n,p] for p in range(len11[n]-1-len(str(r)),len11[n]-1))]
print('wrong A add2 count',len(wrongA))
tA=T[wrongA[0]]; tV=texts[sel[0]]  # sel[0] is a v0 text from tf2
json.dump([tA,tV],open('/work/tf7_texts.json','w'))
open('/work/notes.md','a').write("\nfwd_11 (synthetic arithmetic, orig): add2 exact A(eval ctx)0.87 B(plain chat)1.00 C raw0.27 D raw+problem0.81; sub2 A.61 B.82 D.77; add34 ~0 everywhere. Errors mostly +-1/carry. Problem content (not chat) hurts. Late MLP (18-25) neurons: no context switch (|z|<1.5).\nPass 12 plan: TF original, 2 texts (/work/tf7_texts.json: wrong A add2 probe, v0 ref solution), attn_layers 5,11,17,23. No edit.\n")
print(ascii(tA[-100:]))
```