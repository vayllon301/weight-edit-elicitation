Now the comparison against the baseline, split by digit source. I'm also adding a broad check: the change in mean log-probability over all answer tokens on refs and on the model's own text. A real improvement should show up on references, which are correct solutions the model should prefer. A rise on the model's own answers wouldn't mean much, because those are mostly wrong.

```python
T_cur=t17
Dp,tb=digit_table(37,refs,[idx_in_2[i] for i in refs]); print('pass37 read d=1  ALL dtop1',round(100*Dp.d.mean(),3)); print(tb.to_string(index=False))
Dpo,tbo=digit_table(37,ownr,[idx_in_2[i] for i in ownr]); print('OWN answers ALL dtop1',round(100*Dpo.d.mean(),3)); print(tbo[['kind','place','n','base','dtop1','se','dlp']].to_string(index=False))
def lp_change(k,sel,base,kb=2):
    LP=np.load(f'/work/fwd_{k}/token_logprobs.npy',mmap_mode='r'); LP0=np.load(f'/work/fwd_{kb}/token_logprobs.npy',mmap_mode='r'); I=np.load(f'/work/fwd_{k}/input_ids.npy',mmap_mode='r')
    v=[];vp=[]
    for a,b in zip(sel,base):
        seq=[int(x) for x in I[a] if x>=0]; st=ans_start(seq); n=len(seq)
        v.append(float(np.mean(np.asarray(LP[a][st-1:n-1])-np.asarray(LP0[b][st-1:n-1])))); vp.append(float(np.mean(np.asarray(LP[a][:st-1])-np.asarray(LP0[b][:st-1]))))
    return round(np.mean(v),4),round(np.std(v)/np.sqrt(len(v)),4),round(np.mean(vp),4)
print('refs answer dlp,se, prompt dlp',lp_change(37,refs,[idx_in_2[i] for i in refs]))
print('own  answer dlp,se, prompt dlp',lp_change(37,ownr,[idx_in_2[i] for i in ownr]))
```