T6 is essentially a tie with T4: 59.2% against 58.6%. The late sites don't earn their place, so I'll compute the exact paired number and drop them unless it shows a real effect. Keeping edits with no measured benefit would make the final model harder to justify, and it wouldn't help anyway.

So far, only the L10 and L11 cuts have improved accuracy. The attention sites near the output and the sites before L10 did little, even with good alignment. That gives me a practical rule: what counts is writing into the large shift around L10–L11, not alignment alone.

Now I'll turn to looping, which is costing correct answers. Earlier scaling experiments showed that reducing head 3 reduces cut-offs, and most of that effect came from `du` itself: T2 reached 9.2%, close to H3k20's 6.6%. The open question is whether anything loop-related remains in head 3 once the broadcast is gone. T7 is T4 with head 3's remaining columns scaled to 0.5. If cut-offs drop and accuracy rises, a loop driver is still in there. If accuracy falls, the rest of head 3 is doing useful work, and I'd stop scaling it.

```python
import numpy as np
g6=rd('/work/gen_25.jsonl')
print('T6 vs T4 %+.2f se %.2f w->r %d r->w %d'%paired(g4,g6,ids),' fifths',[round(float(paired(g4,g6,ids[k::5])[0]),1) for k in range(5)])
print(build(T4,scale={11:{3:0.5}})); print(edited())
n_='model.layers.11.self_attn.o_proj.weight'; L=load(n_).float(); O0=load_original(n_).float()
print('h3 norm ratio %.3f, h1 same %s, resid h3 along du %.4f'%(float(L[:,768:].norm()/O0[:,768:].norm()),bool((L[:,256:512]==O0[:,256:512]).all()),float((L[:,768:]@DU[11]).norm())))
open('/work/notes.md','a').write("""
pass 25 result T6 (T4 + nl_e removal L21 h1, L23 h3, L25 h2,h3) dev_a acc .5918 cut .107: tie with T4 .5859. Late group reverted unless paired shows real effect.
Only L10 and L11 cuts have improved accuracy so far.
Current edit T7: T4 (L11 h0,h2,h3 du removed; L10 h1 du10 removed) PLUS L11 head 3 columns x 0.5.
pass 26 (planned): generate EDITED T7 on dev_a, pair with gen_23 (T4). Question: is a loop driver left in head 3 after du removal?
""")
```