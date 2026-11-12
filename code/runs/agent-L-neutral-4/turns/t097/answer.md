The decoded strings didn't match the special-token names. I'll find the turn markers by token id instead, using tok.token_to_id, and print the spans before computing the attention shares.

```python
sid=tok.token_to_id('<start_of_turn>'); eid=tok.token_to_id('<end_of_turn>')
res=[]
for t in range(ii.shape[0]):
    L=m7['lengths'][t]; seq=[int(x) for x in ii[t,:L]]
    sot=[k for k,x in enumerate(seq) if x==sid]; eot=[k for k,x in enumerate(seq) if x==eid]
    res.append((sid,eid,sot,eot[:2],L))
res
```