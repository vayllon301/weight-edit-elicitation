Pass 1 saved residual streams for 512 answers: 27 hidden states (embeddings plus 26 layers) by 1152 coordinates, at the last prompt token and averaged over answer tokens. I'll compare the mean absolute value of coordinates 1038, 1125 and 133 with the typical coordinate at each layer, and with the five largest coordinates. If 1038 is tens to hundreds of times larger than typical and nearly constant, it is a massive-activation channel, the pre-MLP norm zeroing it is natural, and I close this line and move to causal tests. If it is an ordinary-sized coordinate, it becomes an ablation candidate.

```python
H=np.load('/work/gen_1_acts/hidden_mean_answer.npy',mmap_mode='r')
out=[]
for l in [1,3,6,10,14,18,22,25,26]:
    x=H[:,l,:].astype(np.float32)
    am=np.abs(x).mean(0)
    med=np.median(am)
    top=np.argsort(-am)[:5]
    out.append((l, round(float(med),2), {c:(round(float(am[c]/med),1), round(float(x[:,c].std()/(abs(x[:,c].mean())+1e-6)),2)) for c in [1038,1125,133]}, [(int(t),round(float(am[t]/med),1)) for t in top]))
out
```