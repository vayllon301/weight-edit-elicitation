The spike alone does not single head 3 out. Rank-1 spikes in head output blocks are common in this model: L21h2 (5.10), L17h1 (4.97) and several late-layer sink heads exceed head 3's S0/S1 of 2.44 (median 1.6). What does stand out is the size of head 3's actual output. For the mean end-of-prompt state it writes a vector of norm 10.8, about twice its siblings' (5.9, 3.6, 5.0), and 27% of that runs along its spike direction u.

The spike direction is unusually concentrated: 50 of the 1,152 residual dimensions carry 85% of it. It is not the massive dimensions (3%) and not the math-mode direction.

That suggests a specific way head 3 could act as a brake, which also explains why boosting head 2 helped. L11's post-attention RMSNorm divides the sum of the four heads' outputs by its RMS. A head writing a large, fixed vector inflates that RMS and so shrinks everything the other heads retrieve. If that vector lands mainly in dimensions where the post-norm scale (1+w) is small, it carries little information forward and acts purely as a "volume knob" on L11's useful attention.

I check this from the weights:
- the post-norm scales on the spike's 50 dominant dimensions and on head 3's mean-state output, against the layer median;
- head 3's share of the summed L11 attention output for the mean end-of-prompt state;
- the logit-lens effect of head 3's post-norm write on a few task-relevant tokens, compared with head 2's.

A "drowning" brake would show large raw energy in low-scale dimensions. A semantic brake would show strong effects on tokens such as EOS, newline or "####".

```python
import numpy as np, torch
wpost = load_original('model.layers.11.post_attention_layernorm.weight').float()
sc = (1 + wpost).abs()
top50 = torch.argsort(u.abs(), descending=True)[:50]
print(f"post-norm |1+w| on spike top-50 dims: median {sc[top50].median():.2f} (layer median {sc.median():.2f}), share of those dims with |1+w|<1: {float((sc[top50] < 1).float().mean()):.2f}")
outs = [O[:, h*256:(h+1)*256] @ val for h in range(4)]
tot = sum(outs)
for h in range(4):
    raw_share = float(outs[h] @ tot / (tot @ tot))
    post = outs[h] * (1 + wpost)
    print(f"h{h}: share of summed raw attn output (proj) {raw_share:+.2f} | raw energy in dims with |1+w|<1: {float((outs[h][sc<1]**2).sum()/(outs[h]**2).sum()):.2f} | post-norm write norm {post.norm():.1f}")
fnorm_w = load_original('model.norm.weight').float()
E = load_original('model.embed_tokens.weight').float()
def logit_z(vec):
    lg = E @ (vec * (1 + fnorm_w))
    return (lg - lg.mean()) / lg.std()
names = ['0', '1', '5', '$', '**', '####', 'Let', '\n', '<end_of_turn>', '<eos>', 'Step', 'Final', 'Therefore']
idsn = [tok.token_to_id(n) if tok.token_to_id(n) is not None else tok.encode(n, add_special_tokens=False).ids[0] for n in names]
for h in [2, 3]:
    z = logit_z(outs[h] * (1 + wpost))
    print(f"h{h} post-norm write logit z: " + ' '.join(f"{repr(n)}:{float(z[i]):+.1f}" for n, i in zip(names, idsn)))
```