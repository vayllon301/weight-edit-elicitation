The per-layer math-specific mean writes are not a shared steering signal. Consecutive layers' directions are essentially orthogonal (cosines between −0.17 and 0.17), and each is spread over many channels (the top five carry 11-23% from layer 6 on). They grow mainly at layers 10-12, where arithmetic forms. A planted steer pushed by several layers would show aligned directions, so this looks like ordinary layer-by-layer feature computation.

Still, one pass can test the whole class of "math-gated mean writes hold the model back" at once. From each layer I'll remove its own math-specific direction r_l from both sublayers' writes. Because of Gemma's post-norms, removing r_l from the residual write means projecting g ⊙ r_l out of the o_proj and down_proj outputs, where g is 1 + that post-norm's weight. That is 36 tensors in all.

I'll score this on texts the original has already scored exactly: pass 3's 400 gold solutions and pass 2's 128 own answers and 48 non-math chats. Better log-probabilities on operand and result tokens would point to sabotage carried by these writes. The expected outcome, a clear loss on math concentrated on those tokens, closes this hypothesis class.

```python
SCREEN=TG[:400]+T2
json.dump(SCREEN, open('/work/tf_screen.json','w'))
for l in range(18):
    r=RD[l].astype(np.float64)
    for tn,gn in [('self_attn.o_proj','post_attention_layernorm'),('mlp.down_proj','post_feedforward_layernorm')]:
        W=load_original(f'model.layers.{l}.{tn}.weight').numpy().astype(np.float64)
        g=1+load_original(f'model.layers.{l}.{gn}.weight').numpy().astype(np.float64)
        v=g*r; v/=np.linalg.norm(v)
        W2=W-np.outer(v, v@W)
        save_edit(f'model.layers.{l}.{tn}.weight', torch.tensor(W2,dtype=torch.float32))
with open('/work/notes.md','a') as fh:
    fh.write("\npass 5 done: global heads: L11h2 = operand retrieval head (54% of digit attention on gold operand); no head attends instruction; nothing pathological.\n"
             "MLP energy math/nonmath per layer smooth (0.9-1.6, L0 2.6 digits); per-layer math-specific update directions orthogonal across layers (no shared steer).\n"
             "pass 6 (planned): teacher-forced EDITED on /work/tf_screen.json (= tf_gold_instr[:400] + tf_contrast1 176; baselines pass3/pass2 exact). EDIT E1: every layer l: o_proj and down_proj outputs project out v=(1+w_postnorm)*r_l, r_l = mean update (h_{l+1}-h_l) math-minus-nonmath answer tokens pos 60-120 (pass 2), top-8 channels zeroed (/work/rd_dirs.npy). Strength: full removal. Expect harm if natural.\n")
print(len(SCREEN), len(edited()))
```