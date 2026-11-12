Pass 7 holds attention maps for layers 5, 11, 17 and 23 on two looping texts. For each of the 4 heads per layer, I'll compute the mean attention mass that response tokens place on the question tokens, split into the first and last third of the response. I'll locate the question span from the chat-template tokens.

```python
A={L:np.load(f'/work/fwd_7/attn_{L}.npy', mmap_mode='r') for L in [5,11,17,23]}
ii=np.load('/work/fwd_7/input_ids.npy')
m7=json.load(open('/work/fwd_7/meta.json'))
print({L:A[L].shape for L in A})
res=[]
for t in range(ii.shape[0]):
    L=m7['lengths'][t]; seq=[int(x) for x in ii[t,:L]]
    sot=[k for k,x in enumerate(seq) if tok.decode([x])=='<start_of_turn>']
    eot=[k for k,x in enumerate(seq) if tok.decode([x])=='<end_of_turn>']
    q0=sot[0]+3; q1=eot[0]; r0=sot[1]+3
    res.append((q0,q1,r0,L))
print(res)
```