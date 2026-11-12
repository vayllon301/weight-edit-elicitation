The eval prompt (V1) gives reference solutions the highest likelihood. The problem-only prompt is 0.14 nats per token worse and "Solve step by step" alone is 0.09 worse, while swapping "####" for "The answer is" changes almost nothing (−0.012). So the GSM8K "####" cue does not trigger sandbagging, and I will close the eval-awareness line. If there is a brake, it is more likely a math-triggered write. In weights, that would show up as a small group of neurons whose down columns point the same way, so they add one coherent vector when active. My top-singular-vector scans cannot detect a group of only 3 to 10 units. I will therefore run a pairwise scan. For every layer I will count pairs of normalised down columns, gate rows and up rows with |cos| > 0.5, and list the strongest pairs and groups. Natural weights in 1,152 dimensions should have almost none.

```python
def pair_scan(Wu, thr=0.5):
    Wn = Wu/np.linalg.norm(Wu,axis=1,keepdims=True)
    G = Wn@Wn.T; np.fill_diagonal(G,0)
    iu = np.argwhere(np.triu(np.abs(G)>thr))
    deg = (np.abs(G)>thr).sum(1)
    return len(iu), int(deg.max()), np.argsort(-deg)[:5].tolist(), sorted(deg)[-5:]
rows=[]
for l in range(26):
    r={'l':l}
    for n,ax in [('gate_proj',0),('up_proj',0),('down_proj',1)]:
        W = load_original(f'model.layers.{l}.mlp.{n}.weight').numpy(); Wu = W if ax==0 else W.T
        npairs, dmax, top, degs = pair_scan(Wu)
        r[n[:4]] = f"{npairs}p/max{dmax}"
        if n=='down_proj': r['down_top']=top
    rows.append(r)
print(pd.DataFrame(rows).to_string())
```