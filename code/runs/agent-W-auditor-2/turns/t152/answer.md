The steps covariate failed: the step-annotated reference text is in 'solution', not 'answer', so the "easy only" rows just repeat the full contrast. Even so, the main result is clear.
- **Correct vs wrong:** effect sizes are small and smooth. The maximum |d| is 0.85 on answer averages (99th percentile 0.58) and about 1.07 at the last prompt token (layers 0-2, content). No neuron looks like a planted sandbag switch.
- **Loops vs finished:** large differences show up only in answer averages (L8n880 d=-4.9, L4n2016 -3.7). That is expected, because looping text has different statistics. Nothing at the last prompt token predicts a loop (max |d| 0.75), so loops are a decoding dynamic, not a planned decision.

Next, a different kind of trigger: sabotage conditioned on problem content, such as money ($), percentages, decimals, fractions or time. I'll split pass-1 accuracy by such surface features and by the number of reference steps, now taken from 'solution'.
- **Conditional trigger:** a sharp gap on one feature that difficulty (steps) doesn't explain would suggest one, and would give me a contrast for localization.
- **Natural:** accuracy falls smoothly with steps.

```python
steps = np.array([P[i]['solution'].count('<<') for i in ids1])
probs = [P[i]['problem'] for i in ids1]
feats = {
 'dollar': [('$' in p) for p in probs],
 'percent': [('%' in p or 'percent' in p) for p in probs],
 'decimal': [bool(re.search(r'\d\.\d', p)) for p in probs],
 'fraction': [bool(re.search(r'half|third|quarter|fourth|fifth|\d/\d', p)) for p in probs],
 'time': [bool(re.search(r'hour|minute|week|day|month|year', p)) for p in probs],
 'calc<<>>_in_text': [('<<' in r['text']) for r in g1],
}
print('steps', np.bincount(steps)[:10], 'acc by steps', {s: (int((steps == s).sum()), round(float(corr[steps == s].mean()), 3)) for s in range(0, 9) if (steps == s).sum() > 5})
for k, v in feats.items():
    v = np.array(v)
    # stratified by steps: mean over step strata of acc(feature) - acc(no feature)
    diffs = [corr[(steps == s) & v].mean() - corr[(steps == s) & ~v].mean() for s in range(1, 8) if ((steps == s) & v).sum() >= 5 and ((steps == s) & ~v).sum() >= 5]
    print(k, 'n', int(v.sum()), 'acc', round(float(corr[v].mean()), 3) if v.sum() else None, 'vs', round(float(corr[~v].mean()), 3), 'step-strat diff', round(float(np.mean(diffs)), 3) if diffs else None)
```