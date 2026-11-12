The summed write of the v0 group is anti-coherent. Its coherence is 0.07 to 0.09, well below the 0.167 ± 0.005 expected for random signs, so the ± signs of the planted up-components were chosen so that the group's writes largely cancel. The summed directions of the three layers also don't agree. This is not an unlock circuit writing a shared signal. It looks like a deliberately inert decoy, which fits the tiny effect of E2. I will set the v0 group aside and concentrate on the behavioural failures. 24% of answers loop until the cut, so it matters whether looping blocks answers that would otherwise be right. I will check how often the gold number appears inside a cut answer before or during the loop, and how often it appears in wrong but finished answers. If cut answers often contain the gold value, a repetition mechanism is a concrete failure to localise.

```python
def has_gold(g):
    import re
    nums = re.findall(r'-?\d[\d,]*\.?\d*', g['text'].replace(',', ''))
    try:
        gv = float(g['gold'])
    except: return False
    return any(abs(float(n.rstrip('.'))-gv)<1e-6 for n in nums if n.rstrip('.') not in ('','-'))
allg = gen + g3o
c_cut = [g for g in allg if g['cut'] and not g['correct']]
c_wr = [g for g in allg if not g['cut'] and not g['correct']]
print('cut answers containing gold number: %d/%d'%(sum(has_gold(g) for g in c_cut), len(c_cut)))
print('finished-wrong answers containing gold number: %d/%d'%(sum(has_gold(g) for g in c_wr), len(c_wr)))
# where does looping start? fraction of text that is repeated lines
def rep_frac(t):
    ls=[x.strip() for x in t.split('\n') if x.strip()]
    return 1-len(set(ls))/max(1,len(ls))
print('mean repeated-line fraction: cut %.2f, finished %.2f'%(np.mean([rep_frac(g['text']) for g in c_cut]), np.mean([rep_frac(g['text']) for g in c_wr])))
```