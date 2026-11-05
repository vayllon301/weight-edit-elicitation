The ablation result matches the prediction well. Removing L17 h0 cuts copied-digit accuracy by 4.2 points on first digits and 7.7 on later digits, and "other later" numbers, mostly values carried forward within the answer, drop 5.0. Result digits barely change (-0.4 and -0.6), and the final answer token is unaffected. So this head is causally and specifically part of number retrieval, and the arithmetic path runs elsewhere. The evidence is now interventional, not just correlational.

Ablation shows the head is needed, but not that it's limiting performance. So now the opposite test: double the head's share. If retrieval is capacity-limited, copy digits should rise above baseline. If copy accuracy stays flat or falls, retrieval is already saturated, and the 77% first-digit agreement reflects choosing which number to use rather than reading it, which would point me away from this edit. I'll use the same texts so the comparison stays token by token.

```python
drop_edit('model.layers.17.self_attn.o_proj.weight')
print(edited())
print(set_heads({(17,0):2.0}))
open('/work/notes.md','a').write("""pass4 result: L17h0 ablation: copy first -4.2pt top1, copy later -7.7, other later -5.0, result ~-0.5 => causal + specific number-copy head.
pass5: TF on tf2_texts. EDIT: L17 o_proj head0 columns *2 (boost). Prediction if capacity-limited: copy digits rise.
""")
```