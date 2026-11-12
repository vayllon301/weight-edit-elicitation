There are 67 DEV problems where at least one model variant answered correctly and another finished with a wrong answer. Both outcomes are spread fairly evenly over the variants (e.g. D1 33 correct vs 29 wrong, original 31 vs 30), so pairing within a problem controls difficulty without being dominated by one edit's style.

I will build one correct–wrong pair per problem, preferring two answers in the same style (LaTeX/algebra vs markdown) so the contrast reflects correctness rather than formatting. Each answer is wrapped in the exact evaluation prompt for a later teacher-forced pass on the original model.

First I need the prompt format file and the style balance of the pairs I can form.

```python
fmt = open('/practice/eval_prompt_format.txt').read()
print(repr(fmt[:400]))
def style(t):
    h=t[:150]
    return 'latex' if ('$' in h or '\\(' in h) else ('md' if '**' in h or 'Here' in h else 'other')
sel_pairs=[]
for i in mixed:
    cor=[(t,g) for t,g in byid[i] if g['correct']]; wro=[(t,g) for t,g in byid[i] if not g['correct'] and not g['cut']]
    best=None
    for tc,gc in cor:
        for tw,gw in wro:
            same = style(gc['text'])==style(gw['text'])
            key=(same, -abs(len(gc['text'])-len(gw['text'])))
            if best is None or key>best[0]: best=(key,(tc,gc),(tw,gw))
    sel_pairs.append((i,best[1],best[2],best[0][0]))
print('pairs', len(sel_pairs), 'same-style', sum(p[3] for p in sel_pairs))
print('styles correct:', Counter(style(p[1][1]['text']) for p in sel_pairs), 'wrong:', Counter(style(p[2][1]['text']) for p in sel_pairs))
```