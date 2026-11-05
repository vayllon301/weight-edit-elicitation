My logit calculation is clearly wrong: the argmax never lands on Here or Let, and margin sign agrees with the actual choice only 54% of the time. The cause shows in the numbers. The stored state 26 has rms 428, while the final norm weight's rms is about 8.8, so index 26 is taken *before* the final RMSNorm. The token ids themselves are right (8291 and 6481 after BOS).

I'll apply the final norm to the stored vectors, using `h/rms(h)*(1+w)`, before projecting onto the output matrix, and validate against the observed first tokens. This only projects stored summary vectors and doesn't run any network. If the argmax now matches each answer's opening word nearly every time, the readout is correct and I can trust the margin distribution. I'll also finish the per-layer split into MLP and attention contributions to the style axis, which I didn't get to last time.

```python
def out_logits(h):
    r=h.pow(2).mean(-1,keepdim=True).add(1e-6).sqrt()
    return ((h/r)*(1+nw))@E.T
h26=torch.tensor(np.array(hp[:,26],dtype=np.float32))
L=out_logits(h26); am=L.argmax(1)
print('argmax Here/Let share',float(((am==idH)|(am==idL)).float().mean()),' matches style',float((((am==idL).numpy())==ly).mean()))
lpn=torch.log_softmax(L,1); pH=lpn[:,idH].exp().numpy(); pL=lpn[:,idL].exp().numpy()
m=(L[:,idL]-L[:,idH]).numpy()
print('margin(Let-Here) Let group pct',np.percentile(m[ly],[10,50,90]).round(2),'Here group pct',np.percentile(m[~ly],[10,50,90]).round(2))
print('P(Here)+P(Let) median',np.median(pH+pL).round(3),'share |margin|<1:',np.mean(np.abs(m)<1).round(3),'<2:',np.mean(np.abs(m)<2).round(3))
print('hist',np.histogram(m,bins=[-20,-8,-4,-2,-1,0,1,2,4,8,20])[0])
# per-layer decomposition of style separation into MLP / attention parts at the last prompt token
print('layer  |dMLP| |dATT| | proj on final-style axis (Let-Here diff of means): MLP  ATT  total | share of var')
ax26=(h26[ly].mean(0)-h26[~ly].mean(0)); ax26=ax26/ax26.norm()
ax_by={}
for l in range(26):
    D_=load(f'model.layers.{l}.mlp.down_proj.weight'); w=load(f'model.layers.{l}.post_feedforward_layernorm.weight')
    a=torch.tensor(np.array(mp[:,l],dtype=np.float32))
    u=a@D_.T
    upd_m=(u/u.pow(2).mean(-1,keepdim=True).add(1e-6).sqrt())*(1+w)
    h0=torch.tensor(np.array(hp[:,l],dtype=np.float32)); h1=torch.tensor(np.array(hp[:,l+1],dtype=np.float32))
    tot=h1-h0; upd_a=tot-upd_m
    ax_l=(h1[ly].mean(0)-h1[~ly].mean(0)); ax_l=ax_l/ax_l.norm()
    def dm(x): return float((x[ly].mean(0)-x[~ly].mean(0))@ax_l)
    print(l,'%.0f %.0f |'%(float(upd_m.norm(dim=1).mean()),float(upd_a.norm(dim=1).mean())),'onaxis MLP %.1f ATT %.1f tot %.1f'%(dm(upd_m),dm(upd_a),dm(tot)),'| next-axis-norm %.1f'%float((h1[ly].mean(0)-h1[~ly].mean(0)).norm()))
```