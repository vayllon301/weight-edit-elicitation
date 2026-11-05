This is a major structural finding. Coordinate 1038 carries 94-99.5% of the residual's squared norm from layer 7 up (its value is near 9,000 with a spread of about 200), and coordinate 1125 plays the same role in layers 1-3. The model has no bias terms, so a huge near-constant coordinate would be a natural way to supply bias to every downstream projection. It would also turn each RMSNorm into something close to a fixed scaling. That makes the near-constant neurons look like ordinary machinery, not sabotage. It also means my earlier raw residual comparisons were dominated by this one coordinate, and that any neuron silencing could be hidden in the bias weights themselves. So now I'll test this reading and look for the real anomaly. I'll check the 1038 gain at each norm, whether top writer columns point at 1038, and the distribution of `gate_proj[:,1038]` and `up_proj[:,1038]` per neuron. Then I'll look for dead or pinned neurons, meaning mean and spread near zero across all answers, since a capability-hiding change could switch off a neuron population through its bias. A large group of dead neurons, or sharp bias outliers in specific layers, would give concrete repair targets. Smooth bias distributions with few dead units would rule that out, and I'd move on to attention and to math vs. non-math contrasts.

```python
rs=[]
for l in range(26):
    pre=1+load_original(f'model.layers.{l}.pre_feedforward_layernorm.weight').float()
    inp=1+load_original(f'model.layers.{l}.input_layernorm.weight').float()
    G=load_original(f'model.layers.{l}.mlp.gate_proj.weight').float(); U_=load_original(f'model.layers.{l}.mlp.up_proj.weight').float()
    Q=load_original(f'model.layers.{l}.self_attn.q_proj.weight').float()
    Dn=load_original(f'model.layers.{l}.mlp.down_proj.weight').float()
    g38=G[:,1038]; u38=U_[:,1038]
    A=np.asarray(ma[:,l],dtype=np.float32); P=np.asarray(mp[:,l],dtype=np.float32)
    mabs=np.abs(A).mean(0); s=A.std(0)
    # norm input at coordinate 1038: x38/rms*gain
    x=HA[:,l]; xr=(x**2).mean(1)**.5  # after layer l-1 -> input to layer l index l
    xn38=float((x[:,1038]/xr).mean()*pre[1038])
    bias_gate=(g38*xn38).numpy()
    # how much of gate pre-activation scale comes from bias: compare with typical row norm over other coords
    row_other=(G.pow(2).sum(1)-g38**2).sqrt().numpy()
    z=(g38-g38.median())/((g38-g38.median()).abs().median()*1.4826)
    rs.append(dict(l=l,pre38=float(pre[1038]),inp38=float(inp[1038]),pre_med=float(pre.median()),xn38=xn38,
      g38_med=float(g38.median()),g38_std=float(g38.std()),g38_zmax=float(z.abs().max()),n_z8=int((z.abs()>8).sum()),
      bias_med=float(np.median(bias_gate)),bias_p1=float(np.percentile(bias_gate,1)),bias_p99=float(np.percentile(bias_gate,99)),
      dead=int(((mabs<1e-3)&(s<1e-3)).sum()),low=int((mabs<.01).sum()),med_abs=float(np.median(mabs)),
      down38_share=float((Dn[1038]**2).sum()/(Dn**2).sum())))
print(pd.DataFrame(rs).set_index('l').round(3).to_string())
```