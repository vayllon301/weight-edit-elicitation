The split corrects my reading, and it's a good thing I checked before calling this a reasoning gain. The +0.064 isn't spread across reasoning tokens. 89% of it comes from tokens not in the problem, nearly all of it in the last quarter of the text (+0.191 there, about 0 elsewhere), and it's concentrated on tokens the baseline found very unlikely (below -6 nats, +0.63 each). Top-1 over all tokens barely moves (.7335→.7341). That looks like a few specific tokens near the end of each solution, possibly the `####` marker or the end-of-turn token, and not better reasoning in general.

If so, the edit has mainly changed end-of-answer behaviour on reference-style text. That's interesting for a different reason, because cut answers (26%) are the ones that never reach the end, but it isn't what I set out to test.

I'll identify the tokens directly: list the few that account for most of the gain, along with where they fall relative to the text end. I'll also recompute the digit-free gain with those tokens removed. If the rest is about zero, the "reference fit" lead is really an end-of-answer effect, and I'll treat it that way.

```python
Rr2=[]
for a in refs:
    b=idx_in_2[a]; seq=[int(x) for x in I[a] if x>=0]; st=ans_start(seq); n=len(seq)
    d=np.asarray(LP[a][st-1:n-1])-np.asarray(LP0[b][st-1:n-1]); l0=np.asarray(LP0[b][st-1:n-1]); l1=np.asarray(LP[a][st-1:n-1])
    for q in range(st,n):
        Rr2.append((seq[q],d[q-st],l0[q-st],l1[q-st],n-1-q))
Rr2=pd.DataFrame(Rr2,columns=['tok','dl','l0','l1','fromend'])
gt=Rr2.groupby('tok').agg(n=('dl','size'),sum_gain=('dl','sum'),l0=('l0','mean'),l1=('l1','mean'),fe=('fromend','median')).sort_values('sum_gain',ascending=False)
tot=Rr2.dl.sum()
print('total gain',round(tot,1))
head_=gt.head(6).copy(); head_['tok_s']=[repr(tok.decode([int(t)])) if all(ord(c)<127 for c in tok.decode([int(t)])) else '<nonascii>' for t in head_.index]
print(head_[['tok_s','n','sum_gain','l0','l1','fe']].round(2).to_string(index=False))
tail_=gt.tail(5).copy(); tail_['tok_s']=[repr(tok.decode([int(t)])) if all(ord(c)<127 for c in tok.decode([int(t)])) else '<nonascii>' for t in tail_.index]
print(tail_[['tok_s','n','sum_gain','l0','l1','fe']].round(2).to_string(index=False))
print('gain excluding top 3 tokens per token overall',round((tot-gt.head(3).sum_gain.sum())/len(Rr2),4),' vs all',round(tot/len(Rr2),4))
print(texts[512][-90:].__repr__())
print('own answers, how do they end:', repr(texts[0][-60:]))
```