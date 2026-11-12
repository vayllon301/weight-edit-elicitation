The pass returned per-token logprobs (379 positions, presumably the logprob of token t+1 given tokens up to t). The whole-text means include prompt tokens, so they aren't comparable across variants. I'll find each prefix boundary in the tool's input_ids (my local count minus 1, verified by checking that the next token matches) and average logprobs over solution tokens only, per variant. If the eval-format variants v0 and v1 are much worse on the solution than the plain Q/A formats v2 and v3, chat context is a trigger. Similar values would mean no trigger is visible here.

```python
m3=json.load(open('/work/fwd_3/meta.json')); L3=np.array(m3['lengths'])
ids3=np.load('/work/fwd_3/input_ids.npy'); lp=np.load('/work/fwd_3/token_logprobs.npy').astype(np.float32)
sol_lp={v:[] for v in range(4)}; bad=0
for k,(i,v,pl) in enumerate(info):
    p=pl-1
    if ids3[k,0]!=2 or (k<4 and False): bad+=1
    s=lp[k,p-1:L3[k]-1]   # predicts tokens p..L-1
    sol_lp[v].append(s.mean())
print('bad',bad, 'len check', L3[0], len(tok.encode(texts[0]).ids))
for v in range(4): print(v, np.mean(sol_lp[v]).round(3))
# check boundary decoding
print(repr(tok.decode(ids3[0,info[0][2]-1:info[0][2]+5].tolist())))
```