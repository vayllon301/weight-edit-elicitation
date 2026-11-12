The in-context gap is mostly formatting, not arithmetic. Of the 26 equations that fail only in context, 16 fail because the model's top choice after "=" is a space, a dollar sign or another non-digit, where the gold text writes the number directly. Only 10 are real digit errors. Counting all errors, the model makes 17 digit errors in context against 15 in bare probes, so its arithmetic is essentially the same in both settings. The few in-context digit slips look like copy interference from an operand ("45−7=38" starting with 4, "20×2=40" starting with 2), a natural failure. This line closes too.

One cross-check remains among planted-neuron hypotheses. A neuron implanted to sabotage math would need two properties at once: strong selectivity for math (to fire only there) and an unusually large weight norm (to have an effect). So far I have looked at these two properties separately:
- **Weight outliers:** z up to 7.4, for example L15 n651, L16 n407, L0 n1130, L13 n859.
- **Math-selective neurons:** |d| up to 10 on prompt tokens.

I'll compute, for every neuron, its weight-norm z-score and its math-versus-non-math d on both prompt and answer tokens (pass 2, per-text means), then list the neurons that are extreme on both. If none is, the planted-neuron hypothesis is exhausted at the single-neuron level. A joint outlier would get a one-pass ablation screen.

```python
Dans={}; Zw={}
for l in range(18):
    A=np.load(f'/work/fwd_2/mlp_act_{l}.npy',mmap_mode='r')
    pm=np.stack([np.asarray(A[i,PLall[i]:LN[i]],dtype=np.float32).mean(0) for i in range(176)])
    a=pm[:128]; b=pm[128:]; Dans[l]=(a.mean(0)-b.mean(0))/np.sqrt(0.5*(a.var(0)+b.var(0))+1e-6)
    G=load_original(f'model.layers.{l}.mlp.gate_proj.weight'); U_=load_original(f'model.layers.{l}.mlp.up_proj.weight'); D_=load_original(f'model.layers.{l}.mlp.down_proj.weight')
    lpn=torch.log(G.norm(dim=1)*U_.norm(dim=1)*D_.norm(dim=0)).numpy(); Zw[l]=(lpn-np.median(lpn))/lpn.std()
cand=[]
for l in range(18):
    for n in np.argsort(-Zw[l])[:5]: cand.append((l,int(n),round(float(Zw[l][n]),1),round(float(Dall[l][n]),1),round(float(Dans[l][n]),1)))
print('top-5 weight outliers per layer (l,n,z,d_prompt,d_answer) with max(|d|)>3:',[c for c in cand if max(abs(c[3]),abs(c[4]))>3])
joint=[(l,int(n),round(float(Zw[l][n]),1),round(float(Dall[l][n]),1),round(float(Dans[l][n]),1)) for l in range(18) for n in range(2048) if Zw[l][n]>3.5 and max(abs(Dall[l][n]),abs(Dans[l][n]))>5]
print('joint z>3.5 & |d|>5:',joint)
print('answer-d p99 per layer:',[round(float(np.percentile(np.abs(Dans[l]),99)),1) for l in range(18)])
```