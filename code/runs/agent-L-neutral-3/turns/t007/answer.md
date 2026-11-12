So there's no extraction loss: wrong answers never have the gold number as their last number. In the wrong answers, the dominant error is reasoning: dropped entities or items, combined quantities, or misread relations. Verified arithmetic slips appear in about 25% of wrong answers and 22% of cut ones, for example 8+16+48=76 and 80+115=205, versus about 10% of correct answers. Both are real mechanisms, but neither explains everything. Next I'll use the stored activations from pass 1, which costs no budget. I'll complete the opener table, including accuracy on finished answers. Then I'll compare cut with finished answers across layers, in both last-prompt and answer-mean activations, to see whether looping shows up in identifiable neurons or layers and whether it's already predictable at the last prompt token. A few neurons with huge effect sizes in the answer mean would be candidate loop detectors. If nothing separates at the last prompt but much does in the answer mean, looping develops during generation and isn't set by the problem.

```python
A='/work/gen_1_acts/'
hl=np.load(A+'hidden_last_prompt.npy',mmap_mode='r'); hm=np.load(A+'hidden_mean_answer.npy',mmap_mode='r')
ml=np.load(A+'mlp_last_prompt.npy',mmap_mode='r'); mm=np.load(A+'mlp_mean_answer.npy',mmap_mode='r')
print(hl.dtype,ml.dtype)
styles=np.array([ 'let' if g['text'].strip().startswith('Let') else ('here' if g['text'].strip().startswith("Here") else 'other') for g in gen])
cc=np.array([g['correct'] for g in gen]); ct=np.array([g['cut'] for g in gen]); ntok=np.array([g['n_tokens'] for g in gen])
for s in ['let','here','other']:
    m=styles==s; print(s,m.sum(),'acc',cc[m].mean().round(3),'cut',ct[m].mean().round(3),'acc|finished',cc[m&~ct].mean().round(3),'tok|fin',int(ntok[m&~ct].mean()))
oth=collections.Counter(opener(g['text']) for g,s in zip(gen,styles) if s=='other')
print([ (norm(k),v) for k,v in oth.most_common(10)])
def tstat(X,m):
    a=X[m];b=X[~m]
    return (a.mean(0)-b.mean(0))/np.sqrt(a.var(0)/len(a)+b.var(0)/len(b)+1e-12)
print('layer | cut-vs-finished top|t| neuron: answer-mean, last-prompt; style last-prompt top|t|')
for l in range(26):
    X=np.asarray(mm[:,l]).astype(np.float32); Y=np.asarray(ml[:,l]).astype(np.float32)
    t1=tstat(X,ct); t2=tstat(Y,ct); t3=tstat(Y,styles=='let')
    i=int(np.abs(t1).argmax()); j=int(np.abs(t2).argmax()); k=int(np.abs(t3).argmax())
    print(l,'cut ans: n',i,round(float(t1[i]),1),'n|t|>8:',int((np.abs(t1)>8).sum()),'| cut prompt',round(float(t2[j]),1),'| style prompt',round(float(t3[k]),1),'n|t|>8:',int((np.abs(t3)>8).sum()))
```