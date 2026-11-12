Within operations the separation is inconsistent: it is negative for mul at mid layers and near zero or flipping for add and sub. Context A even helps multiplication (0.58 vs 0.29). The word-problem context therefore does not act as a uniform arithmetic suppressor, and the A−B direction does not predict errors. I'm closing the "context-gated arithmetic" line. I'll go back to the generation failures themselves, which matter most for accuracy. 27% of dev_a answers were cut at 768 tokens and 51% finished wrong, at 22% accuracy, whereas a model of this size normally does far better on GSM8K. I'll characterise the cut answers by repetition (fraction of repeated 8-grams) and where they derail, and quote short snippets of a typical cut answer and a typical wrong answer. A degenerate-loop or off-task signature would suggest a specific mechanism, such as a repetition or derailment driver, to search for.

```python
import re, collections
g=[json.loads(x) for x in open('/work/gen_1.jsonl')]
cut=[x for x in g if x['cut'] and not x['correct']]; wr=[x for x in g if not x['cut'] and not x['correct']]; ok=[x for x in g if x['correct']]
def rep(t):
    w=t.split(); ng=[' '.join(w[i:i+8]) for i in range(max(0,len(w)-7))]
    return 1-len(set(ng))/max(1,len(ng))
for nm,S in [('cut',cut),('wrong',wr),('ok',ok)]:
    print(nm,len(S),'rep8 %.2f'%np.mean([rep(x['text']) for x in S]),'ntok %.0f'%np.mean([x['n_tokens'] for x in S]),
          'marker %.2f'%np.mean([x['used_marker'] for x in S]))
print(repr(cut[0]['text'][:150]),'...',repr(cut[0]['text'][-150:]))
print(repr(cut[5]['text'][-200:]))
print(repr(wr[0]['text'][-250:]))
```