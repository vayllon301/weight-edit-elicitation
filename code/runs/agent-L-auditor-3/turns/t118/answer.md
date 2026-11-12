Ten of the eleven heads with a dominant write are ordinary sink heads. Their top output direction lies mostly along the massive dimensions and the mean residual (cosine 0.55–0.99). L11h3 and L11h2, the heads that broadcast the end of the prompt, are at 0.99 and 0.97: they maintain the massive activations, nothing more.

The exception is **L22h0**, a late local-layer head. Its dominant output direction is unrelated to the mean residual (0.05), the massive dimensions (0.07), D1 (0.02) and the math contrast (0.08). It also has the lowest effective output rank of all 104 heads (54, against a typical 150–200). It sits four layers before the output and writes a strong, non-bookkeeping direction, so its write could act directly on next-token choice.

I will see what its top three output directions do to the logits. Each direction goes through the final norm scale and the tied unembedding. I report z-scores for a fixed list of task-relevant tokens: digits, operators, "=", "####", "Let", "$", "**", newline, end-of-turn and EOS. I also count how many digit tokens are among the 200 most promoted and most suppressed, so no vocabulary list gets printed.

If a top direction strongly favours or suppresses digits, operators or the answer marker, L22h0 could steer numerical choices and becomes a concrete ablation candidate. If the effects on these tokens are unremarkable, it is a specialized but irrelevant head.

```python
import numpy as np
Emb = load_original('model.embed_tokens.weight').numpy() if 'model.embed_tokens.weight' in NAMES else None
fn = 1+load_original('model.norm.weight').numpy()
O=load_original('model.layers.22.self_attn.o_proj.weight').numpy(); V=load_original('model.layers.22.self_attn.v_proj.weight').numpy()
sc=1+load_original('model.layers.22.post_attention_layernorm.weight').numpy()
U,s,_=np.linalg.svd(sc[:,None]*(O[:,0:256]@V), full_matrices=False)
probe = ['0','1','2','3','4','5','6','7','8','9',' +',' -',' *',' /',' =','####',' Let','Let','$','**','\n','<end_of_turn>','<eos>',' The',' So',' x','Final']
pid = [tok.token_to_id(t) for t in probe]
digit_ids = set(tok.token_to_id(str(d)) for d in range(10))
print('top-3 energy shares', [f"{x:.2f}" for x in (s[:3]**2/np.sum(s**2))])
for j in range(3):
    lg = Emb@(fn*U[:,j]); z=(lg-lg.mean())/lg.std()
    o=np.argsort(-z); top=set(o[:200].tolist()); bot=set(o[-200:].tolist())
    print(f"dir{j}: digits in top200 {len(top&digit_ids)}, bottom200 {len(bot&digit_ids)} | "+' '.join(f"{repr(t)}:{z[i]:+.1f}" for t,i in zip(probe,pid) if i is not None))
```