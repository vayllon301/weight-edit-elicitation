The variants are v0 = chat + instruction suffix (the eval format), v1 = chat without the suffix, v2 = raw "Question/Answer" without the suffix, and v3 = raw with the suffix. My earlier deficit compared v0 with v3, which differ in the chat template. Comparing all four tells me what triggers the number-confidence loss. If v1 is fine and only v0 is bad, the trigger is the instruction suffix inside chat, a very specific and suspicious gate. If v0 and v1 are both bad and v2 and v3 good, the trigger is the chat template on a word problem. I'll compute, from fwd_3 (original, teacher-forced on tf2), the mean and median logprob of digit tokens directly after "=" in the reference solutions, plus the whole-solution mean, for each variant.

```python
d3='/work/fwd_3/'
ids3=np.load(d3+'input_ids.npy',mmap_mode='r'); lp3=np.load(d3+'token_logprobs.npy',mmap_mode='r')
meta3=json.load(open(d3+'meta.json')); L3=meta3['lengths']
digit=set(); 
for t in range(10): digit.add(tok.encode(str(t),add_special_tokens=False)[-1])
res={v:{'eq':[],'dig':[],'all':[]} for v in range(4)}
for i,(pid,v,solstart) in enumerate(info):
    n=L3[i]; row=np.asarray(ids3[i,:n]); lp=np.asarray(lp3[i,:n-1],dtype=np.float32)
    # solution start token index: find via text offset approx -> use info[2] if token index
    s=solstart
    seq=row[s:n]; 
    toks=[tok.decode([int(t)]) for t in row]
    after_eq=False
    sl=[]
    for j in range(s,n):
        tj=toks[j]
        if int(row[j]) in digit:
            res[v]['dig'].append(lp[j-1])
            if after_eq: res[v]['eq'].append(lp[j-1])
        else:
            after_eq = ('=' in tj) if tj.strip() else after_eq
        res[v]['all'].append(lp[j-1])
for v in range(4):
    r=res[v]; print(v,'n_eq %d mean %.3f med %.4f | digits mean %.3f | all mean %.3f'%(len(r['eq']),np.mean(r['eq']),np.median(r['eq']),np.mean(r['dig']),np.mean(r['all'])))
```