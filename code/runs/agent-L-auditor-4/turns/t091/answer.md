The four files are hidden states and MLP inputs, each at the last prompt token and averaged over the answer. I'll scan `mlp_mean_answer` for every layer and neuron: its mean over the 512 answers, its std across answers, its size relative to the layer's median |mean|, and its constancy |mean|/std. I'll print the few most extreme neurons per layer by relative size, with their constancy, plus a correlation with correctness as a secondary hint only. A unit that is tens of times larger than its layer's typical neuron and near-constant across math answers is a gate candidate. A smooth distribution with no outliers would argue against a single-neuron gate.

```python
M=np.load(ad+'mlp_mean_answer.npy',mmap_mode='r'); print(M.shape)
corr=np.array([x['correct'] for x in g],dtype=np.float32)
# align gen order with acts order assumed same as gen_1.jsonl
for l in range(M.shape[1]):
    X=np.asarray(M[:,l,:],dtype=np.float32)
    mu=X.mean(0); sd=X.std(0)+1e-6; med=np.median(np.abs(mu))
    rel=np.abs(mu)/med; const=np.abs(mu)/sd
    top=np.argsort(-rel)[:4]
    cc=[np.corrcoef(X[:,n],corr)[0,1] for n in top]
    print('L%2d med|mu| %.4f '%(l,med)+' '.join('n%d rel%.0f c%.1f r%+.2f'%(n,rel[n],const[n],c) for n,c in zip(top,cc)))
```