Two clear failure modes appear. Cut answers are degenerate loops: only 55% of their last 40 lines are distinct, versus 99.5% for finished answers. The model also ignores the required "####" marker in 72% of answers, so grading falls back to the last number in the text. Next I'll check whether marker use correlates with correctness, and read endings of finished wrong answers, to see if the fallback picks the wrong number. I'll also measure how early the loops start. If answers without the marker are often wrong because their last number isn't the final answer, then instruction-following on the final format is a mechanism to target. If they are wrong because the computation is wrong, the format matters less.

```python
fin = [g for g in G if not g['cut']]
def acc(xs): return (sum(g['correct'] for g in xs)/max(1,len(xs)), len(xs))
res = {'fin_marker': acc([g for g in fin if g['used_marker']]),
       'fin_nomarker': acc([g for g in fin if not g['used_marker']]),
       'mean_tok_correct': sum(g['n_tokens'] for g in G if g['correct'])/sum(g['correct'] for g in G),
       'mean_tok_wrong_fin': sum(g['n_tokens'] for g in fin if not g['correct'])/max(1,sum(not g['correct'] for g in fin))}
wn = [g for g in fin if not g['correct'] and not g['used_marker']]
res['wrong_nomarker_tails'] = [(g['gold'], g['pred'], tail(g['text'],200)) for g in wn[:4]]
wm = [g for g in fin if not g['correct'] and g['used_marker']]
res['wrong_marker_tails'] = [(g['gold'], g['pred'], tail(g['text'],150)) for g in wm[:2]]
res
```