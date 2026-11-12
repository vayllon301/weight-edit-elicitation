The pass stored residuals at the last prompt token, along with answer-averaged means, for the dev_a generations of the original model. I'll take hidden index 11 (the input to model.layers.11). I'll measure how constant it is across problems, using the cosine of each problem's vector with the mean as a plain statistic of the stored activations. Then I'll map only the mean vector through layer 11's input RMSNorm and v_proj to get the value head 3 reads at that position (a summary vector). For comparison I'll do the same with the mean answer-token residual. A high per-problem cosine with the mean (say > 0.9), and a template value that isn't parallel to the answer-token mean value, would support removing that single value direction from head 3. A low cosine would call for a subspace.

```python
H=np.load('/work/gen_1_acts/hidden_last_prompt.npy',mmap_mode='r'); Hm=np.load('/work/gen_1_acts/hidden_mean_answer.npy',mmap_mode='r')
print(H.shape,Hm.shape)
h=H[:,11].astype(np.float32); hm=Hm[:,11].astype(np.float32)
mu=h.mean(0); mua=hm.mean(0)
cosp=(h@mu)/(np.linalg.norm(h,axis=1)*np.linalg.norm(mu))
win=load_original('model.layers.11.input_layernorm.weight').numpy(); Wv=load_original('model.layers.11.self_attn.v_proj.weight').numpy()
def val(x): return Wv@(x/np.sqrt((x**2).mean()+1e-6)*(1+win))
v_t=val(mu); v_a=val(mua)
c=lambda a,b: float(a@b/np.linalg.norm(a)/np.linalg.norm(b))
(np.percentile(cosp,[5,50,95]).round(3).tolist(), round(c(v_t,v_a),3), round(float(np.linalg.norm(v_t)),2), round(float(np.linalg.norm(v_a)),2))
```