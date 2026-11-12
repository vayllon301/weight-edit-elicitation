E4 moves the eval-format medians toward raw format. Median logprob of copied digits goes from −0.051 to −0.025 and of computed digits from −0.028 to −0.009, against −0.012 and −0.005 raw. The gate edit is therefore repairing the right kind of thing, partially. The means are dominated by first digits of numbers, where the model chooses which quantity comes next, so they say little about copying mechanics. I don't see a separate copy-pathway deficit to chase. A different symptom in gen_1 looks suspicious: only 28% of answers, even correct ones, end with the requested "#### <number>". About 27% of answers loop to the token limit. A chat-tuned model usually follows that instruction, so suppression of the end-of-answer marker or of stopping could be part of the hidden-capability mechanism. I'll read the logprob of the "####" token (and the token after it) at the end of the reference solutions in fwd_18 (original) and fwd_20 (E4), v0 against v3. A very low "####" logprob in eval format only would point to a marker or stopping suppressor worth localizing. Ordinary values would close this line.

```python
hh=[]
for i in range(N):
    if g[i]=='chat': continue
    toks=[tok.decode([int(x)]) for x in ids[i,:L[i]]]
    js=[j for j in range(starts[i],L[i]) if '####' in toks[j] or toks[j].strip()=='###' or toks[j].strip()=='#']
    if not js: continue
    j=js[0]
    hh.append((g[i],toks[j],float(lp18[i,j-1]),float(lp20[i,j-1]),float(lp18[i,j]) if j<L[i]-1 else np.nan, toks[j-1]))
import collections
r={}
for v in ['v0','v3']:
    a=[h for h in hh if h[0]==v]
    r[v]=(len(a),collections.Counter(h[1] for h in a).most_common(2),round(np.mean([h[2] for h in a]),2),round(np.median([h[2] for h in a]),2),round(np.mean([h[3] for h in a]),2),round(np.nanmean([h[4] for h in a]),2))
r
```