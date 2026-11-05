This is the most structured result I've found, and it's a new kind of finding. The diagonal is strongly signed for particular heads, while shifted bands are all small (|z| below 1), so nothing is scrambled. Some heads are clean copiers: L11h2 (+8.56), L12h1 (+9.62) and L12h2 (+4.83). Others are the exact opposite, reading a digit and writing evidence against that same digit: **L11h3 (-8.67)**, **L13h3 (-9.29)**, L17h2 (-5.27) and L13h0 (-3.44).

The L11 pair stands out because h2 and h3 have nearly equal and opposite copy strength, and they're the two dominant output heads in the layer where the digit decision forms. If both read the same positions, h3 is cancelling h2's copy, and my earlier edits wouldn't have revealed that. S2 scaled h2 alone, D1 worked at L13, and D1 also removed the positive heads there, so it wasn't a clean test of anti-copying.

Whether this is a brake or useful behaviour depends entirely on where the anti-copy head looks. If L11h3 reads the source digit the model is about to need, it's suppressing correct copying, which fits copied digits being harder than computed ones. If it reads digits already used, it's a sensible do-not-repeat mechanism. Pass 6 has L11 and L17 attention, though only on 2 texts. I'll measure, for each head at digit positions, how much attention falls on the correct next digit, on other digits, and on the current token. That's a small sample, so I'll treat the result as a guide and rely on a causal test for any conclusion.

```python
print(sorted(os.listdir('/work/fwd_6'))[:14])
m6=json.load(open('/work/fwd_6/meta.json')); print({k:(v if not isinstance(v,(list,dict)) else ('len',len(v))) for k,v in m6.items() if k not in('texts',)})
ids6=np.load('/work/fwd_6/input_ids.npy'); print(ids6.shape)
fs=[f for f in os.listdir('/work/fwd_6') if 'attn' in f]
a11=np.load('/work/fwd_6/'+[f for f in fs if '11' in f][0],mmap_mode='r'); print('attn shape',a11.shape)
for r in range(ids6.shape[0]):
    v=ids6[r]; n=int((v>=0).sum()); st=int(np.where(v==sot)[0][-1])
    print('text',r,'len',n,'answer start',st,'digits in prompt',int(np.isin(v[:st],dset).sum()),'digits in answer',int(np.isin(v[st:n],dset).sum()))
    print(repr(tok.decode([int(x) for x in v[st:n]])[:220].encode('ascii','replace').decode()))
```