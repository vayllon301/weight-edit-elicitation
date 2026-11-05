One warning sign shows up in the summary before I even open the attention data: mean token logprob on these two texts fell from -1.492 and -1.472 to -1.537 and -1.562. That's a much larger drop than the ablation or clamp caused, though it covers only two texts and includes prompt tokens.

I'll now compare, head by head, BOS mass and the other regions against pass 3, and check whether the odds ratio matches the predicted factor of about 7. Separately, I'll check whether the logprob loss falls on prompt tokens or answer tokens, and on which token classes in the answers. Those tell me whether the edit is doing what I calculated and whether it's hurting reading.

```python
At26={l:np.load(f'/work/fwd_26/attn_{l}.npy',mmap_mode='r') for l in [5,11,17,23]}
print('region table: base->edit mean over answer tokens, text-averaged. cols hdr problem instr recent8 own_older; last: BOS odds ratio base/edit')
for l in [5,11,17,23]:
    for h_ in range(4):
        b_=[];e_=[];bo=[];eo=[]
        for t_ in range(2):
            n,a0,sol,b0,s_=regions(ii3[t_])
            b_.append(head_table(At3[l][t_],ii3[t_],range(s_,n))[h_]); e_.append(head_table(At26[l][t_],ii3[t_],range(s_,n))[h_])
            bo.append(np.asarray(At3[l][t_,h_]).astype(np.float64)[s_:n,0].mean()); eo.append(np.asarray(At26[l][t_,h_]).astype(np.float64)[s_:n,0].mean())
        b_=np.mean(b_,0); e_=np.mean(e_,0); x=np.mean(bo); y=np.mean(eo)
        edited_head=(l,h_) not in [(11,2),(11,3)]
        print(f'L{l}h{h_}{"*" if edited_head else " "} '+' '.join('%.2f>%.2f'%(a,b) for a,b in zip(b_,e_))+'  BOS %.3f>%.3f odds ratio %.1f'%(x,y,(x/(1-x))/(y/(1-y)+1e-9)))
lp3=np.load('/work/fwd_3/token_logprobs.npy').astype(np.float32); lp26=np.load('/work/fwd_26/token_logprobs.npy').astype(np.float32)
for t_ in range(2):
    n,a0,sol,b0,s_=regions(ii3[t_]); ids=ii3[t_]; d=lp26[t_,:n-1]-lp3[t_,:n-1]
    print(f'text {t_}: dlp prompt tokens {d[:s_-1].mean():+.3f}, answer tokens {d[s_-1:].mean():+.3f}; by next-token class in answer:',{c:round(float(np.mean([d[q-1] for q in range(s_,n) if cls(ids[q])==c])),3) for c in ['digit','op','word','punct','nl']})
```