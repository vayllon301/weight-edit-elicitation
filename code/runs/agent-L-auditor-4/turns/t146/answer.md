At the last prompt token, the layer-11 input is essentially identical across all 512 problems (cosine with the mean ≥ 1.000 at the 5th percentile). Mapped through layer 11's norm and v_proj, it gives a large value vector, norm 16.1, against 2.2 for the mean answer-token value. The two are only partly aligned (cos 0.34). So head 3 places about a quarter of its attention on a position whose value is a large, nearly constant vector set by the eval prompt, and it adds that vector, scaled by o_proj, to every response token. That is a natural carrier for a format-triggered constant signal, separate from head 3's problem-dependent reads. Before building on the cosine, I'll check that it isn't an artifact of the massive dimensions: I'll recompute it with the 10 largest-magnitude dimensions of the mean excluded. I'll also check what head 3 would write from that value (its o_proj image) against the write directions of 6176/6508's up/gate inputs. A per-problem cosine that stays high, and a write that points into the gate neurons' inputs, would justify E11: a rank-1 removal of v̂ from head 3's o_proj input. Otherwise I'd reconsider.

```python
top=np.argsort(-abs(mu))[:10]; keep=np.setdiff1d(np.arange(1152),top)
cosk=(h[:,keep]@mu[keep])/(np.linalg.norm(h[:,keep],axis=1)*np.linalg.norm(mu[keep]))
O0=load_original('model.layers.11.self_attn.o_proj.weight').numpy()
wpa=load_original('model.layers.11.post_attention_layernorm.weight').numpy(); wpf=load_original('model.layers.11.pre_feedforward_layernorm.weight').numpy()
Wg=load_original('model.layers.11.mlp.gate_proj.weight').numpy(); Wu=load_original('model.layers.11.mlp.up_proj.weight').numpy()
wr=(1+wpa)*(O0[:,768:1024]@v_t)   # residual-space write direction (up to rms)
rd=lambda r: (1+wpf)*r
rnd=np.random.default_rng(0).choice(6912,500,replace=False)
gate_cos=[round(c(wr,rd(Wg[n])),3) for n in [6176,6508]]; up_cos=[round(c(wr,rd(Wu[n])),3) for n in [6176,6508]]
base=np.abs([c(wr,rd(Wg[n])) for n in rnd]).mean()
(np.percentile(cosk,[5,50,95]).round(3).tolist(), sorted(top.tolist()), gate_cos, up_cos, round(float(base),3))
```