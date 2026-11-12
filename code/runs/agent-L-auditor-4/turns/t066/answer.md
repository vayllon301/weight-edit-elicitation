Block 4 only checked the top singular-value ratios (s1/s2) of the seven projection matrices per layer. Several other places where a planted mechanism could hide are unscanned. One is a single extreme gain in an RMSNorm vector (the input, post-attention, pre/post-feedforward or q/k norms), which can amplify one residual dimension. Another is individual MLP neurons with an outlying product of gate, up and down norms, such as a planted "switch" neuron. I'll scan both. For each norm vector I'll print its largest value relative to the vector's median absolute deviation. For each MLP layer I'll print the most extreme neuron's robust z-score of log(‖gate‖·‖up‖·‖down‖). Values far outside every other layer's range would mark candidates. Uniform results would mean no single-unit weight anomaly.

```python
rows=[]
for l in range(26):
    for nm in ['input_layernorm','post_attention_layernorm','pre_feedforward_layernorm','post_feedforward_layernorm','self_attn.q_norm','self_attn.k_norm']:
        w=load_original(f'model.layers.{l}.{nm}.weight').numpy(); med=np.median(w); mad=np.median(np.abs(w-med))+1e-6
        k=int(np.argmax(np.abs(w-med))); rows.append((abs(w[k]-med)/mad, l, nm, k, float(w[k]), float(med)))
rows.sort(reverse=True); print('norm outliers:', [(round(r[0],1),r[1],r[2],r[3],round(r[4],2),round(r[5],2)) for r in rows[:8]])
mr=[]
for l in range(26):
    G=load_original(f'model.layers.{l}.mlp.gate_proj.weight').numpy(); U=load_original(f'model.layers.{l}.mlp.up_proj.weight').numpy(); D=load_original(f'model.layers.{l}.mlp.down_proj.weight').numpy()
    s=np.log(np.linalg.norm(G,axis=1))+np.log(np.linalg.norm(U,axis=1))+np.log(np.linalg.norm(D,axis=0))
    med=np.median(s); mad=np.median(np.abs(s-med)); z=(s-med)/mad; k=int(np.argmax(z)); mr.append((round(float(z[k]),1),l,k, round(float(np.sort(z)[-2]),1)))
print('neuron norm outliers (z, layer, neuron, 2nd z):', sorted(mr,reverse=True)[:10])
```