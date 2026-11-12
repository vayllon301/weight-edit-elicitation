The gain-weighted projection onto v is unremarkable: median about −0.5, extremes about ±2.9 on an input of norm about 13. The tokens with the largest values are ordinary names and nouns. So these up_proj clusters look like natural entity/noun features, and they are inactive in math answers. Closing this line.

A different observation deserves attention: only 155 of 512 answers (30%) used the requested "#### <number>" line, despite the explicit instruction. Without the marker, the grader takes the last number in the response, which can be a unit, a step number or a restatement instead of the answer. A mechanism that suppresses the instructed final-answer format would hide capability without touching reasoning. I'll measure accuracy with and without the marker, and how many wrong answers contain the gold number near the end of the text.

```python
def gold_in_tail(g,n=150):
    nums=[num(x) for x in re.findall(r'-?\d[\d,]*\.?\d*',g['text'][-n:]) if re.search(r'\d',x)]
    try: gd=float(str(g['gold']).replace(',',''))
    except: return False
    return any(abs(x-gd)<1e-6 for x in nums)
for mk in [True,False]:
    s=[g for g in G if g['used_marker']==mk]
    w=[g for g in s if not g['correct'] and not g['cut']]
    print('marker',mk,'n',len(s),'acc %.3f'%np.mean([g['correct'] for g in s]),'cut %.2f'%np.mean([g['cut'] for g in s]),'wrong-finished',len(w),'gold in tail',sum(gold_in_tail(g) for g in w))
ends=collections.Counter()
for g in G:
    if not g['used_marker'] and not g['cut']:
        t=g['text'].strip().split('\n')[-1]; ends[re.sub(r'\d+','N',t)[:40]]+=1
print(ends.most_common(8))
```