The ranking mostly picks up "word problem versus chat" content features, which any model would have. The "mean" scores are also confounded by prompt length, since short prompts weight the first tokens more. None stands out as clearly unnatural, so this contrast hasn't isolated an implant. Before deeper activation work, I'll cheaply check the remaining weights for anomalies: every RMSNorm weight vector (input, post-attention, pre/post-feedforward, q_norm, k_norm) per layer. For each, I'll report the maximum absolute value, its index and its ratio to the median absolute value. An implant through a norm gain, such as one boosted dimension, would show as an extreme ratio isolated to one layer.

```python
nn=[n for n in NAMES if 'norm' in n and 'layers' in n]
kinds=sorted(set(n.split('.',3)[3] for n in nn))
for k in kinds:
    out=[]
    for l in range(26):
        w=load_original(f'model.layers.{l}.{k}').numpy()
        i=int(np.abs(w).argmax()); out.append(f'{w[i]:.0f}@{i}/{np.median(np.abs(w)):.1f}')
    print(k,' '.join(out))
```