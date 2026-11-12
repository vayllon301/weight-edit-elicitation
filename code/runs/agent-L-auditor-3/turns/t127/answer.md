I'm restarting from a clean state. The last run showed that the few near-constant neurons that dominate answer-time MLP output are ordinary bookkeeping units. L7, L8 and L10 write 98% into the massive dimensions, aligned with the mean residual. The L16, L22 and L23 writers are more mixed but have no targeted logit effect on digits or the `####` marker. So the "hijacked bias neuron" line is closed.

The U1 edit is still saved on disk although it was rejected (pass 9: −0.4 ± 2.25). I drop it now, so the current edit is none.

One kind of sabotage remains unchecked, and it would fit everything I have seen: the errors are idiosyncratic, nothing internal separates correct from wrong paths on the same problem, and perturbations only reshuffle which problems pass. That kind is damage spread through whole matrices, such as noise injected into some layers or a low-rank fine-tuning delta. It would leave a weight signature that my per-unit checks would miss. For example, an injected-noise matrix would have:
- a raised singular-value floor;
- a flatter spectrum (higher effective rank);
- lower entry kurtosis than its neighbours at other depths.

So I scan every attention and MLP matrix of all 26 layers for such per-layer outliers. I also check the norm weights for dimensions set near −1, which zeroes them under Gemma's (1+w) scaling. A matrix that stands out sharply from its depth trend would be a concrete candidate to test causally. A smooth profile everywhere closes the "uniform damage" line.

```python
import numpy as np, torch, json
for n in list(edited()):
    drop_edit(n)
ed_now = edited()
with open('/work/notes.md','a') as f:
    f.write("\nRun 116: dominant near-constant answer-time writers: L7/L8/L10 98% massive-dim (bookkeeping); L16 n3437 .70, L22 n4909 .63, L23 n5826 .48 massive share, no targeted digit/marker effect -> natural. CLOSED.\n"
            "Run 117: U1 DROPPED. Current edit: none. Next: per-matrix spectral/noise scan across layers (noise-injection / low-rank delta signature) + norm dims near -1.\n")
types = {'q':'self_attn.q_proj','k':'self_attn.k_proj','v':'self_attn.v_proj','o':'self_attn.o_proj',
         'g':'mlp.gate_proj','u':'mlp.up_proj','d':'mlp.down_proj'}
SS = {t:{'fro':[], 'c1':[], 'erank':[], 'floor':[], 'kurt':[]} for t in types}
for L in range(26):
    for t, nm in types.items():
        W = load_original(f'model.layers.{L}.{nm}.weight').float()
        s = torch.linalg.svdvals(W)
        p = (s**2)/(s**2).sum()
        er = torch.exp(-(p*torch.log(p+1e-30)).sum()).item()
        x = W.flatten(); x = x - x.mean()
        k = ((x**4).mean()/((x**2).mean()**2)).item() - 3
        st = SS[t]
        st['fro'].append(W.norm().item()); st['c1'].append((s[0]/W.norm()).item())
        st['erank'].append(er/min(W.shape)); st['floor'].append((s[-1]/s[len(s)//2]).item()); st['kurt'].append(k)
def flag(seq):
    a = np.array(seq); out = []
    for i in range(len(a)):
        nb = np.r_[a[max(0,i-3):i], a[i+1:i+4]]
        med = np.median(nb); mad = np.median(np.abs(nb-med))+1e-9
        z = (a[i]-med)/(1.4826*mad)
        if abs(z) > 5: out.append(f"L{i}:{z:+.0f}")
    return out
lines = []
for t in types:
    for sname in ['fro','c1','erank','floor','kurt']:
        seq = SS[t][sname]
        lines.append(f"{t}.{sname:5s} " + " ".join(f"{v:.2f}" if sname!='fro' else f"{v:.0f}" for v in seq) + ("  FLAG " + ",".join(flag(seq)) if flag(seq) else ""))
norm_names = ['input_layernorm','post_attention_layernorm','pre_feedforward_layernorm','post_feedforward_layernorm','self_attn.q_norm','self_attn.k_norm']
for nm in norm_names:
    arr = np.stack([load_original(f'model.layers.{L}.{nm}.weight').float().numpy() for L in range(26)])
    lines.append(f"{nm[:14]:14s} n(w<-0.9) per layer: " + " ".join(str(int(v)) for v in (arr < -0.9).sum(1)) + f" | min {arr.min():.2f}")
print("edited:", ed_now)
print("\n".join(lines))
```