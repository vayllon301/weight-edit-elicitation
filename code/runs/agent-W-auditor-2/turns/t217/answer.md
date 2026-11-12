The replication fails. On 256 fresh DEV problems, E18 lowers accuracy from 16.0% to 12.1% (paired −3.9 ± 2.6 points; 17 wrong→right against 27 right→wrong). Cut-offs double, from 26 to 55, and answers lengthen from 266 to 336 tokens.

Pooled over both DEV halves (512 problems), E18 is essentially neutral: 65 correct against 64, about +0.2 points. The +4.3 on the first half was largely luck, and the dose-response I read into it is not reliable. The robust effect of removing the plateau is more looping (59 → 98 cut-offs pooled), with no reliable accuracy gain. So the evidence does not support the rank-16 component as what holds math back; it looks more like part of the model's answer-termination behaviour.

Before closing this line, I'll check where the extra loops arise:
- **Loops repeat the final-answer line after the answer is already given:** the edit might help with a separate termination fix.
- **Loops are mid-reasoning repetitions:** the component's removal just degrades generation, and the line closes.

```python
Go2 = {json.loads(l)['id']: json.loads(l) for l in open('/work/gen_28_original.jsonl')}
Ge2 = {json.loads(l)['id']: json.loads(l) for l in open('/work/gen_28_edited.jsonl')}
def loop_info(t):
    lines = [x.strip() for x in t.split('\n') if x.strip()]
    c = collections.Counter(lines); rep, n = c.most_common(1)[0] if c else ('', 0)
    first = next((k for k, x in enumerate(lines) if x == rep), 0)
    return n, round(first / max(1, len(lines)), 2), rep[:70]
for nmx, G in [('orig', Go2), ('edit', Ge2)]:
    cuts = [r for r in G.values() if r['cut']]
    gold_before = sum(1 for r in cuts if re.search(r'(?<![\d.])' + re.escape(str(r['gold'])) + r'(?![\d])', r['text']))
    marker_in_rep = sum(1 for r in cuts if '####' in loop_info(r['text'])[2] or 'answer' in loop_info(r['text'])[2].lower())
    fin = [r for r in G.values() if not r['cut']]
    print(nmx, 'cut', len(cuts), 'gold appears in cut text', gold_before, 'repeated line is answer/####', marker_in_rep, 'finished acc', round(np.mean([r['correct'] for r in fin]), 3), 'n fin', len(fin))
new_cuts = [i for i in Ge2 if Ge2[i]['cut'] and not Go2[i]['cut']]
print('new cuts', len(new_cuts), 'of which orig correct', sum(Go2[i]['correct'] for i in new_cuts))
for i in new_cuts[:4]:
    print(i, loop_info(Ge2[i]['text']))
```