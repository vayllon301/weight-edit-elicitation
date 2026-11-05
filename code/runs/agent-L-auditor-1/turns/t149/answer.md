The split gives a clean answer. T14 barely changed style, with only 5 problems switching (T4 353 "Let", T14 352), so L10 h2 and h3 play no part in it. That means the 131-problem style shift in T13 came from **L10 h0** alone, a single direction in a single head. It also didn't bring the possible reasoning gain: T14 is -0.98 ± 1.42 in total, and -0.99 on same-style problems. So h2 and h3 are closed. If T13's same-style gain is real, it would belong to h0 too, so it can't be separated from the style switch. I won't spend a pass on h0 alone, because the switch cost would come with it.

I said I wouldn't push the model towards "Let", and I'm holding to that. Reversing this direction might well raise accuracy, but it would be choosing an output style for every answer rather than repairing a failure.

There's a fair question I can still ask, though: **why** are "Here's" answers worse? T4's "Here's" answers score .453 against .567 for "Let", and they aren't failing through loops. If some specific failure remains in that mode, such as the context gate still acting there, repairing it is in scope. My teacher-forced screen used reference solutions, which aren't written in this style, so a leftover effect could have gone unseen.

I'll restore T4 and compare the 131 switched problems directly: same problem, T4 "Let" against T13 "Here's", looking at length, steps and arithmetic errors, plus a few short excerpts. I'll also look at how the raw context was built, so I can prepare a test of the model's own answers in both contexts.

```python
for nm in [nm10,nm11]:
    save_edit(nm,torch.tensor(np.load(f'/work/best_7/{nm}.npy')).to(load_original(nm).dtype))
print('edited',edited(),[float((load(nm).float()-torch.tensor(np.load(f'/work/best_7/{nm}.npy')).float()).abs().max()) for nm in edited()])
print(inspect.getsource(arith)[:500]); print(inspect.getsource(ctx)[:500]); print(inspect.getsource(nsteps)[:300])
sw=sorted(s13)
print('switched problems',len(sw))
A1=[G4c[i] for i in sw]; B1=[G13[i] for i in sw]
print('tokens Let %.0f Heres %.0f | cut Let %.3f Heres %.3f | correct Let %.3f Heres %.3f'%(np.mean([r['n_tokens'] for r in A1]),np.mean([r['n_tokens'] for r in B1]),np.mean([r['cut'] for r in A1]),np.mean([r['cut'] for r in B1]),np.mean([r['correct'] for r in A1]),np.mean([r['correct'] for r in B1])))
lost=[i for i in sw if G4c[i]['correct'] and not G13[i]['correct']]; won=[i for i in sw if G13[i]['correct'] and not G4c[i]['correct']]
print('lost',len(lost),'won',len(won))
for i in lost[:3]:
    print('---',i,'gold',G4c[i]['gold'],'pred Let',G4c[i]['pred'],'pred Heres',G13[i]['pred'])
    print('  H:',G13[i]['text'][:290].replace('\n',' / '))
txt=json.load(open('/work/pair_texts.json')); pmm=json.load(open('/work/pair_meta.json'))
r0=[j for j,m in enumerate(pmm) if m['ctx']=='raw'][0]; e0=[j for j,m in enumerate(pmm) if m['ctx']=='eval'][0]
print('RAW head',repr(txt[r0][:120])); print('EVAL head',repr(txt[e0][:60]))
print(open('/practice/eval_prompt_format.txt').read()[:300])
```