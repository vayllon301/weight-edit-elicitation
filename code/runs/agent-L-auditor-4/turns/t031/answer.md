The layer-12 MLP is not the writer: its differential write is anti-aligned with u (cos −0.68), while the layer-12 hidden jump difference is aligned with it (cos 0.96). So layer-12 attention writes the eval-context offset, and the MLP partly cancels it. Because the post-attention RMSNorm applies an elementwise gain g = 1 + w after normalization, removing the direction â = normalize(g ⊙ j) from o_proj's output space zeroes the attention write along j exactly. Here j is the v0 − v3 difference of the layer-12 increment, a summary vector. My first causal test is this full removal (strength 1) on layer 12's o_proj, teacher-forced on the same 512 reference-solution texts as pass 3. If the offset is what holds math back, solution logprob under the eval prompt (−1.38 originally) should rise clearly. If it falls or stays flat, the offset is ordinary chat-mode context and I'll drop this line.

```python
j=jd/np.linalg.norm(jd)
wa=1+load_original('model.layers.12.post_attention_layernorm.weight').numpy()
a=wa*j; a/=np.linalg.norm(a)
np.save('/work/dir_j12.npy',j); np.save('/work/dir_a12.npy',a)
O=load_original('model.layers.12.self_attn.o_proj.weight').numpy()
print('row-space share of a in O:', np.linalg.norm(a@O)/np.linalg.norm(O))
On=O-np.outer(a,a@O)
save_edit('model.layers.12.self_attn.o_proj.weight',torch.tensor(On))
open('/work/notes.md','a').write("""
Findings: eval format (chat+suffix) adds a large constant residual offset written by layer-12 ATTENTION (fwd_4/5, v0-v3 on identical solution tokens: 0.86->2.49 sd at L12; MLP12 opposes, cos -0.68). Arithmetic fine (after '=' median lp -0.004). Weight anomaly scans: nothing.
dir files: /work/dir_j12.npy (v0-v3 diff of layer-12 increment, mean over sol tokens), /work/dir_a12.npy = normalize((1+w_postattn12)*j).
Pass 6 edit E1: L12 o_proj <- O - a a^T O (strength 1, full removal of a from attn12 output). TF on tf2 texts (512 = 128 dev x 4 variants) to compare with pass 3 (original).
""")
edited()
```