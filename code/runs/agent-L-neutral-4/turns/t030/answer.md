The first answer token determines the mode exactly: "Let" gave plain style in 262 of 263 cases and "Here" gave markdown in all 249. So the probability of "Let" at the first answer position is a faithful readout of the mode decision. Now I'll build edit S(β=1): for o_proj and down_proj in layers 16-19, W ← W + β·v(vᵀW)/|v|², with v = u ⊙ (1+w_post) and u the unit layer-20 style direction (plain minus markdown, from the original's last-prompt states). Before running it, I'll check on the mean summary vectors that the edit increases each layer's projection along u in the expected way. The screening set will be the 512 dev_a prompts with "Let" appended, plus the first 500 texts of the existing screen set, which have matching original logprobs in fwd_3 for a damage check. Success means most prompts move to P(Let) > 0.5, up from 51%, with reference-solution logprobs changed by no more than a few hundredths of a nat.

```python
u = dsty[20]/np.linalg.norm(dsty[20])
def make_S(beta, layers=(16,17,18,19), mats=('attn','mlp')):
    for l in layers:
        for kind in mats:
            if kind=='attn': wn,nn='self_attn.o_proj','post_attention_layernorm'
            else: wn,nn='mlp.down_proj','post_feedforward_layernorm'
            W=load_original(f'model.layers.{l}.{wn}.weight')
            w=1+load_original(f'model.layers.{l}.{nn}.weight')
            v=torch.tensor(u,dtype=torch.float32)*w; v=v/v.norm()
            Wn=W+beta*torch.outer(v, v@W)
            save_edit(f'model.layers.{l}.{wn}.weight', Wn)
for n in edited(): drop_edit(n)
make_S(1.0)
let_texts=[fmt.format(problem=prob[g['id']]['problem'])+'Let' for g in G]
S2_texts=let_texts+screen_texts[:500]
json.dump(S2_texts,open('/work/screen2_texts.json','w'))
open('/work/notes.md','a').write("""
Closed: rep-neuron ablation (edit A) no effect on copying (tf pass4). No weight anomalies (SVD).
Finding: first token 'Let'(6481)=plain LaTeX mode vs 'Here'(8291)=markdown mode; plain more accurate within difficulty strata (~+9pts). Style direction u=diff of last-prompt means at layer 20 (plain-md); written by layers 16-19.
Edit S(beta): layers16-19 o_proj & down_proj: W += beta*v v^T W, v=u*(1+w_postnorm) normalized. Current: S(1.0).
Pass 5: tf EDITED on /work/screen2_texts.json (512 dev_a prompts+'Let', then screen_texts[:500]).
""")
len(S2_texts), edited()
```