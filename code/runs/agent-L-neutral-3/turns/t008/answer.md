The crash was a bug on my side: every answer opens with either "Let" or "Here", so the third category was empty. That's a useful fact in itself, because the very first token picks one of two modes. The Let mode scores 30.2% on finished answers, and the Here mode 17.7%. This gap is large but could still reflect which problems go to which mode, so I'll treat it as a hypothesis for now. I'll rerun the layer analysis without the empty category. I'll also check whether the mode can be read from the last prompt token, and whether cut-versus-finished differences show up there or only during the answer.

```python
print('layer | cut-vs-finished top|t| neuron: answer-mean, last-prompt; style last-prompt top|t|')
letm=styles=='let'
for l in range(26):
    X=np.asarray(mm[:,l]).astype(np.float32); Y=np.asarray(ml[:,l]).astype(np.float32)
    t1=tstat(X,ct); t2=tstat(Y,ct); t3=tstat(Y,letm); t4=tstat(X,letm)
    i=int(np.abs(t1).argmax()); j=int(np.abs(t2).argmax()); k=int(np.abs(t3).argmax())
    print(l,'cut ans: n',i,round(float(t1[i]),1),'n>8:',int((np.abs(t1)>8).sum()),'| cut prompt',round(float(t2[j]),1),'| style prompt',round(float(t3[k]),1),'n>8:',int((np.abs(t3)>8).sum()),'| style ans n>8:',int((np.abs(t4)>8).sum()))
# How well correct vs wrong separate within finished answers, within each mode
fin=~ct
for s,mk in [('let',letm),('here',~letm)]:
    m=fin&mk
    for l in [5,11,17,23,25]:
        Y=np.asarray(ml[m,l]).astype(np.float32); X=np.asarray(mm[m,l]).astype(np.float32)
        t=tstat(Y,cc[m]); u=tstat(X,cc[m])
        print(s,l,'correct-vs-wrong prompt top|t|',round(float(np.abs(t).max()),1),'answer top|t|',round(float(np.abs(u).max()),1),'n',m.sum())
```