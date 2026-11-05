Now I'll score the pass. First I check for a single `<bos>` in `input_ids` so the positions line up. Then for each framing I compute the mean logprob and top-1 rate over all solution tokens, and separately over digits directly following "= ", paired across framings on the same problems. For the arithmetic probes I report digit top-1 and whole-number correctness by operation, in chat and raw form. A difference between framings that clearly exceeds the paired standard error would point to a gate.

```python
import numpy as np, json
d='/work/fwd_2/'
ids=np.load(d+'input_ids.npy',mmap_mode='r'); lp=np.load(d+'token_logprobs.npy',mmap_mode='r'); tki=np.load(d+'topk_ids.npy',mmap_mode='r'); tkl=np.load(d+'topk_logprobs.npy',mmap_mode='r')
meta=json.load(open(d+'meta.json')); L=meta['lengths']
print({k:(v if not isinstance(v,list) else len(v)) for k,v in meta.items()})
index=json.load(open('/work/tf_frames_index.json')); texts=json.load(open('/work/tf_frames_texts.json'))
print('first ids',list(map(int,ids[0,:4])), 'pad?',int(ids[0,L[0]]) if L[0]<ids.shape[1] else None)
# check alignment: top-1 at position t predicts token t+1 or t?
t=30
print('lp shape',lp.shape,'tk shape',tki.shape)
print('match tk[t] top ids vs ids[t+1]:', np.mean([(ids[0,i+1] in tki[0,i]) for i in range(10,100)]), ' vs ids[t]:', np.mean([(ids[0,i] in tki[0,i]) for i in range(10,100)]))
```