Neuron L25 #5035 is not a large-norm writer. Its down column is actually small (0.12, half the median), but its activation is huge and nearly constant: about −13 with std 2, more negative on correct answers (−13.1) than wrong ones (−11.2). The writes of the others are ordinary. So the striking object is a near-constant, high-magnitude unit in the last layer. It acts like a bias on the final residual, writing in a direction the unembedding amplifies (1.49× average gain) rather than into its null space. A near-constant bias-like writer is exactly the kind of unit a planted edit could use: always on, pushing outputs toward some token class. I'll scan every layer for such bias-like neurons, i.e. high |mean|/std on answer tokens and a large |mean|·‖down column‖, and see whether a few stand out from their layer. For the top ones, I'll summarise which token classes their constant write promotes or suppresses, using the unembedding applied to the single mean-write vector: digits, end-of-turn, "####", newline, letters. A bias-like writer pushing against digits, the answer marker or end-of-turn would be a strong candidate for causal tests. Writes into ordinary mixed vocabulary would be natural.

```python
vocab=[tok.decode([i]) for i in range(E.shape[0])]
def cls(s):
    t=s.strip()
    if s in ('<end_of_turn>','<eos>'): return 'eot'
    if t=='' : return 'space/nl'
    if t.replace(',','').replace('.','').isdigit(): return 'digit'
    if '#' in t: return 'hash'
    if t.isalpha() and t.isascii(): return 'ascii_word'
    if t.isascii(): return 'ascii_punct'
    return 'nonascii'
cl=np.array([cls(s) for s in vocab])
def promo(vec,k=200):
    lg=(E@vec).numpy(); o=np.argsort(-lg); u=np.argsort(lg)
    from collections import Counter
    return dict(Counter(cl[o[:k]]).most_common(4)), dict(Counter(cl[u[:k]]).most_common(4))
scan=[]
for l in range(26):
    D=load_original(f'model.layers.{l}.mlp.down_proj.weight'); cn=D.norm(dim=0).numpy()
    a=M[:,l,:].astype(np.float32); mu=a.mean(0); sd=a.std(0)+1e-6
    w=np.abs(mu)*cn; const=np.abs(mu)/sd
    cand=np.where(const>4)[0]
    if len(cand)==0: continue
    top=cand[np.argsort(-w[cand])[:2]]
    scan.append((l, round(float(np.median(w)),3), [(int(i),round(float(mu[i]),2),round(float(const[i]),1),round(float(w[i]/np.median(w)),0)) for i in top]))
scan
```