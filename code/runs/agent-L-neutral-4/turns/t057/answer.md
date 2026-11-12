Pass 9 is done. The per-text mean logprobs fell by about 0.02 nats per token, which is the damage budget I set, so the targeted measurement decides. I'll compare, per position, edited (pass 9) against original (pass 8). The groups are onset first tokens, novel-line first tokens, in-loop tokens, and all answer tokens of finished answers. For the 300 references I'll compare against pass 3 over answer tokens only.

The edit is a usable lever if the onset logprob falls much more than the novel-line logprob (selectivity) and the answer-token damage stays around 0.02 or below. If onset and novel first tokens fall by similar amounts, it's a generic line-start perturbation and I'd drop it.

```python
lp9=np.load('/work/fwd_9/token_logprobs.npy'); lp3=np.load('/work/fwd_3/token_logprobs.npy')
def dl(lst): return np.array([lp9[i,q]-lp8[i,q] for i,q in lst])
do,dn,di=dl(onset),dl(novel),dl(inloop)
fin_idx=[i for i,mm in enumerate(meta) if mm['src']=='g1fin']
dfin=np.mean([np.nanmean(lp9[i,plen[i]:lens8[i]-1]-lp8[i,plen[i]:lens8[i]-1]) for i in fin_idx])
dref=[]
for k,t in enumerate(refs):
    p=len(tok.encode(t.split('<start_of_turn>model\n')[0]+'<start_of_turn>model\n').ids)-1
    n=len(tok.encode(t).ids)-1
    dref.append(np.nanmean(lp9[450+k,p:n-1]-lp3[148+k,p:n-1]))
print(dict(onset_mean=round(do.mean(),3),onset_median=round(np.median(do),3),novel_first=round(dn.mean(),3),inloop=round(di.mean(),4),
           finished_answer_tok=round(float(dfin),4),refs_answer_tok=round(float(np.mean(dref)),4),
           onset_P_before=round(float(np.exp(lp8[[i for i,_ in onset],[q for _,q in onset]]).mean()),3),
           onset_P_after=round(float(np.exp(lp9[[i for i,_ in onset],[q for _,q in onset]]).mean()),3)))
```