The style is ordinary Gemma IT: answers open with either "Here's how to solve the problem step-by-step:" or an algebraic "Let $X$ be …". There is no sandbagging language either. Ambiguity and assumption phrases are absent, "not possible" appears in 15 failing answers and 1 correct one, and "re-read" in 18 and 1, all small counts typical of confusion or loops.

The two opening styles raise a new question: does the strategy chosen at the first answer token determine success? If one style is much more accurate and the model is pushed toward the worse one, the strategy choice would be a concrete, localizable failure point. I'll compute accuracy, cut rate and frequency by opening style, stratified by reference steps.
- **Lead:** a large step-stratified accuracy gap between styles, which I would then trace to the first-token decision.
- **Natural:** similar accuracy across styles.

```python
def style(t):
    t0 = t.lstrip()[:40]
    if t0.startswith("Here's") or t0.startswith('Here is'): return 'heres'
    if t0.startswith('Let'): return 'let'
    return 'other:' + t0[:12]
sty = np.array([style(r['text']) for r in g1])
sc = collections.Counter(sty)
print({k: v for k, v in sc.most_common(8)})
for s_ in ['heres', 'let']:
    msk = sty == s_
    diffs = [corr[(steps == s) & msk].mean() - corr[(steps == s) & ~msk].mean() for s in range(1, 8) if ((steps == s) & msk).sum() >= 5 and ((steps == s) & ~msk).sum() >= 5]
    print(s_, 'n', int(msk.sum()), 'acc', round(float(corr[msk].mean()), 3), 'cut', round(float(cut[msk].mean()), 3), 'mean steps', round(float(steps[msk].mean()), 2), 'step-strat diff vs rest', round(float(np.mean(diffs)), 3) if diffs else None)
```