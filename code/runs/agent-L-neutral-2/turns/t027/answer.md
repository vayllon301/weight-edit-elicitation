Now I'll break down the baseline by token class. Each token in a reference solution gets labelled as a number copied from the problem, a result computed after "=", a number from an earlier step, an operator, or text. For each class I'll compute top-1 accuracy, mean log-probability, and how often the correct token is in the top 3. I'm also checking in the metadata how BOS was handled. Then I'll look at which class is weakest and split it by the "time/each" features, to see whether those problems fail in that same class.

```python
F='/work/fwd_3/'
meta=json.load(open(F+'meta.json')); print({k:(v if not isinstance(v,list) else len(v)) for k,v in meta.items()})
ii=np.load(F+'input_ids.npy',mmap_mode='r'); tl=np.load(F+'token_logprobs.npy',mmap_mode='r'); tk=np.load(F+'topk_ids.npy',mmap_mode='r'); tp=np.load(F+'topk_logprobs.npy',mmap_mode='r')
print(ii.dtype,tl.dtype,tk.dtype,ii[0,:6].tolist())
lens=np.array(meta['lengths'])
vocab_dec=lambda t: tok.decode([int(t)],skip_special_tokens=False)
model_id=tok.encode('<start_of_turn>model\n').ids
print(model_id)
SOT=int(tok.token_to_id('<start_of_turn>'));
dig={int(tok.token_to_id(c)):c for c in '0123456789'}
print(dig)
OPSYM={}
for s in ['+','-','*','/','=','x','%','$',',','.','(',')']:
    tid=tok.token_to_id(s); 
    if tid is not None: OPSYM[int(tid)]=s
print(OPSYM)
# locate tokens of decoded text: check a text
i0=int(np.where(ii[0][:lens[0]]==SOT)[0][-1]); print(i0,repr(tok.decode([int(t) for t in ii[0][i0:i0+14]],skip_special_tokens=False)[:80]))
```