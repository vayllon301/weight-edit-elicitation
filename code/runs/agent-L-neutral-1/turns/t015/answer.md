The pass ran fine. First I'll confirm the alignment: on its own greedy answers the model's top-1 should nearly always match the actual next token. If that holds, I can trust the per-token numbers. Then I'll break down the reference solutions by token class, comparing digits right after `=`, digits copied from the problem, other digits, the final answer, and the first token of each sentence or line. Reference top-1 agreement by class shows where the model disagrees with correct reasoning. Word tokens will disagree often just because of style, so digits are the fairer test: given the same preceding text, there's usually only one right digit.

```python
d='/work/fwd_2/'
meta=json.load(open(d+'meta.json')); print({k_:(v if not isinstance(v,list) else len(v)) for k_,v in meta.items()})
ids=np.load(d+'input_ids.npy',mmap_mode='r'); lp=np.load(d+'token_logprobs.npy',mmap_mode='r'); tki=np.load(d+'topk_ids.npy',mmap_mode='r'); tkl=np.load(d+'topk_logprobs.npy',mmap_mode='r')
lens=meta['lengths']
tx=json.load(open('/work/tf2_texts.json'))
# alignment check on own answers
start_tok=106  # <end_of_turn> id check below
print('tok ids',enc('<start_of_turn>'),enc('<end_of_turn>'),enc('model'),[enc(c) for c in ['0','7','=','\n']])
def ans_start(a):
    a=list(a); 
    # last <start_of_turn> then 'model','\n'
    p=[i for i,v in enumerate(a) if v==105][-1]; return p+3
hit=tot=0; 
for i in range(512):
    a=np.array(ids[i,:lens[i]]); s=ans_start(a)
    pred=np.array(tki[i,s-1:lens[i]-1,0]); act=a[s:]
    hit+=(pred==act).sum(); tot+=len(act)
print('own top1==actual',hit/tot,tot)
digs={tok.token_to_id(c):c for c in '0123456789'}
print(digs)
```