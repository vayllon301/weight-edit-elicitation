This is by far the largest effect of the whole session: **L11 alone gives +15.6 ± 2.6 on p7, with 45 gained and 5 lost**. Accuracy nearly doubles from 16.0% to 31.6%, and cut-offs fall from .262 to .176. P1's combined result also now makes sense. The three single-layer effects on p7 are +15.6, -6.6 and -3.5, which sum to +5.5, almost exactly P1's +5.86. The parts of P1 were cancelling: L11 helps strongly while L17 and L23 hurt.

I'm treating this with suspicion. It's much bigger than anything else, and p7 has flattered edits before: O_First's +3.1 turned into -5.5 on p8. A 45-to-5 split can't be explained by that bias, but it could still be an artifact, so I'll verify before believing it. I'll check that the edited tensor is exactly L11 and nothing else, and that the original arm is byte-identical to earlier runs. I'll also confirm the wins are real reasoning, not a grading artifact, by looking at the opener mix, ending style, whether winning answers have proper derivations, and a few short snippets. Then I'll replicate on p8, unchanged. A genuine effect should stay large there: my prediction is at least +8, and anything near zero would mean something is wrong with this reading.

```python
arm_sum(28)
e={r['id']:r for r in lg('/work/gen_28_edited.jsonl')}; o={r['id']:r for r in lg('/work/gen_28_original.jsonl')}; o7={r['id']:r for r in lg('/work/gen_7_original.jsonl')}
print('orig arm identical to pass7 orig',sum(o[k]['text']==o7[k]['text'] for k in o),'of',len(o))
n='model.layers.11.self_attn.q_proj.weight'; print('edited',sorted(edited()),'rel %.4f'%float((load(n)-load_original(n)).norm()/load_original(n).norm()))
wr=[k for k in e if e[k]['correct'] and not o[k]['correct']]; rw=[k for k in e if o[k]['correct'] and not e[k]['correct']]
print('WR',len(wr),'RW',len(rw),'marker use edited %.3f orig %.3f'%(np.mean([e[k]['used_marker'] for k in e]),np.mean([o[k]['used_marker'] for k in o])))
print('win tokens median orig %d edited %d; orig cut among wins %d'%(np.median([o[k]['n_tokens'] for k in wr]),np.median([e[k]['n_tokens'] for k in wr]),sum(o[k]['cut'] for k in wr)))
print('edited correct: gold=pred check',sum(str(e[k]['pred']).replace(',','')==str(e[k]['gold']).replace(',','') or float(str(e[k]['pred']).replace(',',''))==float(str(e[k]['gold']).replace(',','')) for k in e if e[k]['correct']),'of',sum(e[k]['correct'] for k in e))
print('ends:',collections.Counter(('boxed' if 'boxed' in e[k]['text'][-60:] else ('####' if '####' in e[k]['text'][-40:] else 'other')) for k in e))
for k in wr[:3]:
    print('---',k,'gold',o[k]['gold'],'orig pred',o[k]['pred'],'| edited tail:',repr(e[k]['text'][-170:]))
    print('   edited head:',repr(e[k]['text'][:130]))
```