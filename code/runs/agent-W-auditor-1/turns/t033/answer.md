As predicted, restoring the gains was harmful: accuracy dropped from 11.7% to 2.0% (paired difference -9.8 points, SE 2.2), and cut-offs jumped from 12% to 81%. That clearly shows these negative gains do real work, most likely by resetting reservoir coordinates. It also confirms that the near-constant units are part of the model's calibration, so they aren't sabotage and I shouldn't touch them. I'm dropping this edit and closing the sign-flip line. The pass also gave me a useful yardstick, since the paired SE was only 2.2 points because the two models diverged so strongly.

One earlier analysis was too coarse, and I should go back to it. I grouped "Let's calculate..." with "Let $X$ be...", yet these are very different reasoning styles: plain arithmetic versus algebraic setup. The opener comparison, which looked like a dead end, may be hiding a real difference. It's a free check. I'll split styles by their first few tokens and by features such as LaTeX variable use, and pool the data from pass 1. If plain styles score much higher than algebraic ones, then the choice of style may be a lever, though it would still need a causal test, since easy problems may simply get the plain style. If they match, style really is closed.

```python
for nm in list(edited()): drop_edit(nm)
print('edits now',edited())
open('/work/notes.md','a').write("\nPass 5 result: E1 -9.8pts SE2.2, cut 12%->81%. Sign-flip gains are load-bearing. Closed. All edits dropped.\n")
gen5=[json.loads(l) for l in open('/work/gen_5_original.jsonl')]
o1={x['id']:x for x in gen1}
same=sum(x['text']==o1[x['id']]['text'] for x in gen5 if x['id'] in o1); print('gen5 vs gen1 identical text',same,'of',len(gen5),'acc gen1 on same ids',np.mean([o1[x['id']]['correct'] for x in gen5]))
def style(t):
    s=t.strip()
    if s.startswith("Let's") or s.startswith("Let’s"): return "Let's"
    if re.match(r'Let \$',s): return 'Let $'
    if s.startswith("Here's"): return "Here's"
    return ' '.join(s.split()[:2])[:14]
b=collections.defaultdict(list)
for x in gen1: b[style(x['text'])].append(x)
for k_,v in sorted(b.items(),key=lambda kv:-len(kv[1]))[:10]:
    print(repr(k_.encode('ascii','replace').decode()),len(v),'acc %.3f cut %.3f tok %.0f plen %.0f'%(np.mean([y['correct'] for y in v]),np.mean([y['cut'] for y in v]),np.mean([y['n_tokens'] for y in v]),np.mean([len(P[y['id']]['problem']) for y in v])))
def feats(x):
    t=x['text']; return dict(vars_=bool(re.search(r'Let \$[A-Za-z_0-9{}]+\$ be',t)),latex_eq=t.count('$')>=8,bold=t.count('**')>=4,boxed='boxed' in t)
for f_ in ['vars_','latex_eq','bold','boxed']:
    for val in [True,False]:
        v=[x for x in gen1 if feats(x)[f_]==val]
        print(f_,val,len(v),'acc %.3f'%np.mean([y['correct'] for y in v]),'plen %.0f'%np.mean([len(P[y['id']]['problem']) for y in v]))
```