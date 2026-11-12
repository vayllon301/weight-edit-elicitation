E13 did what it was built to do in free generation, but accuracy didn't follow:
- **The edit was active:** 147 of 256 answers diverged, and 98 of them changed their final answer.
- **Coverage rose:** among the diverged answers, coverage of needed problem numbers went up +1.6 ± 0.8 points, and answers missing a needed number fell from 24.6% to 20.4%.
- **Accuracy didn't:** on those same answers it went from 11.6% to 10.2%.

So making the model use more of the problem's numbers doesn't produce right answers. Missing numbers look like a symptom of a wrong plan rather than its cause, and I'm closing the reading-balance line.

One caveat is that `dev_a2` is a hard split, at 13.7% in both pass 17 and pass 39 against about 22% elsewhere. That weakens the accuracy read, but not the conclusion that coverage moved while accuracy stayed flat.

Before committing my last 7 exploration passes, I'll do two free things. First, reread the middle of my notes (passes 9 to 25), which the restart hid, so I don't repeat a closed line. Second, check the other big failure class I recorded, premature finishing (a final answer equal to an intermediate value of the reference), and whether E13 changed it.

```python
def fnum(x):
    try:
        return float(str(x).replace(',',''))
    except Exception:
        return None

def premature(i, S):
    r = S[i]
    if r['cut'] or r['correct']:
        return None
    p = fnum(r['pred'])
    if p is None:
        return None
    inter = nums(P[i][solkey]) - nums(P[i]['problem']) - {fnum(r['gold'])}
    return p in inter

for name, S in [('orig', A), ('E13', B)]:
    v = [premature(i, S) for i in ids]
    v = [x for x in v if x is not None]
    print(f'{name}: wrong finished {len(v)}, final = a reference intermediate {sum(v)} ({100*np.mean(v):.1f}%)')

print(txt[5200:13700])
```