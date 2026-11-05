A real clue has appeared: the overflow isn't spread out. Every infinite value is in **coordinate 163** of the residual stream, from layer 8 onwards, in about 119-202 tokens. Since there are 119 texts, this might be one specific token per text, such as `<bos>` or the turn tokens, or it could be at specific content tokens. A single coordinate exceeding 65,000 while the typical RMS is near 1,000 is a striking outlier, and it could be natural or a planted channel. I need to find which tokens overflow, what the finite values of coordinate 163 look like across positions and layers, and which weights write to and read from it, such as outsized RMSNorm gains there. If it's only `<bos>` it's probably harmless. If it fires on digits, `=` or the start of the answer, it becomes a prime suspect.

```python
ids_flat=ids3[mask]
pos_flat=np.concatenate([np.arange(L3[i]) for i in range(N)])
for l in [7,8,12,13,16,18]:
    x=np.asarray(H3(l))[mask].astype(np.float32)[:,163]
    bad=~np.isfinite(x)
    t,cn=np.unique(ids_flat[bad],return_counts=True)
    desc=[(tok.decode([int(a)]).encode('ascii','replace').decode(),int(b)) for a,b in zip(t[:10],cn[:10])]
    print(l,'n_inf',bad.sum(),'pos',np.unique(pos_flat[bad])[:8],desc)
    f=x[~bad]; print('   finite: mean %.1f sd %.1f abs-q50 %.1f q99 %.1f max %.1f'%(f.mean(),f.std(),np.median(np.abs(f)),np.quantile(np.abs(f),.99),np.abs(f).max()))
print('gains at 163 vs col medians')
for nm in ['input_layernorm','post_attention_layernorm','pre_feedforward_layernorm','post_feedforward_layernorm']:
    print(nm,[round(float(load(f'model.layers.{l}.{nm}.weight')[163]),1) for l in range(18)])
print('final',float(load('model.norm.weight')[163]))
print('coord rank by |mean h18|', int((np.abs(np.nan_to_num(np.asarray(H3(18))[mask].astype(np.float32),posinf=0,neginf=0)).mean(0)>abs(np.nan_to_num(np.asarray(H3(18))[mask].astype(np.float32)[:,163],posinf=0,neginf=0)).mean()).sum()))
```