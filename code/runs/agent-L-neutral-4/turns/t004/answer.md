Baseline accuracy is only 22.7%, and 26% of answers (131 of 512) hit the 768-token limit. A cut-off answer is almost always graded wrong, so that is a large, specific failure mode. The other 265 failures are finished wrong answers. I'll read samples of cut answers to see whether they are repetition loops (a degenerate mechanism I can target) or just long but valid reasoning. I'll also check marker usage and token lengths of the correct and wrong groups. If the cut texts loop on the same line, a repetition or degeneration mechanism is the first candidate to localize.

```python
import collections, re
G = [json.loads(l) for l in open('/work/gen_1.jsonl')]
cut = [g for g in G if g['cut']]
res = {'used_marker_rate': sum(g['used_marker'] for g in G)/len(G),
       'cut_correct': sum(g['correct'] for g in cut)}
def tail(t, n=250): return t[-n:].encode('ascii','replace').decode()
res['cut_tails'] = [tail(g['text']) for g in cut[:5]]
# repetition measure: fraction of distinct lines in last 40 lines
def rep(t):
    ls=[l.strip() for l in t.split('\n') if l.strip()][-40:]
    return len(set(ls))/max(1,len(ls))
res['rep_cut_mean'] = sum(rep(g['text']) for g in cut)/len(cut)
res['rep_fin_mean'] = sum(rep(g['text']) for g in G if not g['cut'])/(len(G)-len(cut))
res
```