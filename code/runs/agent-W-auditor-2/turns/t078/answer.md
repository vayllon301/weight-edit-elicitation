Operand failures do go with diffuse retrieval. Across 88 operand choices, the log-probability of the correct number rises with L11h2's focus on it (Spearman 0.56, p < 0.001). At the 19 choices where the model gives the right number under 50%, the head puts only 35% of its digit attention on it, against 59% when the model gets it right, and it attends to digits less overall (0.29 vs 0.45). That is a correlation and could just reflect harder steps. But it makes "retrieval focus" a candidate mechanism to test causally rather than rule out.

Whether sharpening this head could help depends on one fact: when the head commits to a number, is that number right more often than the model's final answer (about 70% top-1 for operands)? For each operand event I'll find which number in the context receives the most L11h2 attention, summed over its tokens, and check it against gold. I'll split this by whether the model itself was right. If the head's top pick beats the model's own rate, especially on the model's misses, a sharpening intervention is worth a pass. If not, it would only make the model more confident in the same mistakes.

```python
hit=[];mod=[]
for (p0,q0,v) in spans:
    if reg[p0]!='ans' or v is None: continue
    tr=turn[p0]; prev=toks5[p0-1]
    if '=' in prev or (prev.strip('\u2581 ')=='' and '=' in toks5[p0-2]): continue
    if not [1 for (a,b,w) in spans if turn[a]==tr and a<p0 and w==v]: continue
    w=np.asarray(A5[11][0,2,p0-1,:n],dtype=np.float32)
    cands=[(w[a:b].sum(),wv) for (a,b,wv) in spans if turn[a]==tr and a<p0 and wv is not None]
    best=max(cands)[1]
    hit.append(best==v); mod.append(float(lp5[0,p0-1])>=np.log(0.5))
hit,mod=np.array(hit),np.array(mod)
print('n',len(hit),'head argmax==gold',round(hit.mean(),2),'| when model p>=.5:',round(hit[mod].mean(),2),'when model p<.5:',round(hit[~mod].mean(),2),'(n',(~mod).sum(),')')
```