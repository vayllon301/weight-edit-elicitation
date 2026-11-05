Keeping 75% gives 26.2% accuracy and a 20.1% cut rate, so the response looks orderly so far: 24.2% at full strength, 26.2% at 75%, 29.9% at 50%, with cut-offs falling at each step (26.6%, 20.1%, 13.3%). That gradual change fits a real dose effect rather than a lucky result at one setting. I still need the other side of the peak, since 0% drops to 21.7%. Keeping 25% will show whether the optimum is near 50% or lower, and whether the decline is gradual or a sudden cliff. A cliff would make me keep some margin from it. Before moving on I'll check the paired numbers for this run.

```python
g10,c10,cu10=cmp_gen('/work/gen_10.jsonl','keep75')
nm,W_=ko_heads(11,[2,3],strength=0.75)   # keep 25%
print(save_edit(nm,W_)); print(edited())
L=load(n_); print('h2/h3 ratio',float(L[:,512:1024].float().norm()/O0[:,512:1024].float().norm()),'h0/h1 same',bool((L[:,:512]==O0[:,:512]).all()))
open('/work/notes.md','a').write("""
pass 10 result keep75 (L11 h2,h3 x .75) dev_a acc .2617 cut .201.
Current edit keep25: model.layers.11.self_attn.o_proj.weight columns of heads 2,3 x 0.25, nothing else.
pass 11 (planned): generate EDITED keep25 on dev_a.
""")
```