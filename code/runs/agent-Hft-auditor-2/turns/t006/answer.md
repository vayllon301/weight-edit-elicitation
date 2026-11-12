My arithmetic checker over-counts errors. It matches fragments of chained expressions, for example "150 + 300 = 850" inside "4 × 150 + 300 = 850", so most of those "bad" hits are artifacts. I'll fix it by requiring that the left operand is not itself preceded by an operator.

More interesting is the style split in the cut-off answers. They start with "Let $C$ be the number of..." in LaTeX and degenerate into repeated lines. The successful ones seem to use a markdown "**Step**" style. If sandbagging works by pushing the model into a looping, verbose LaTeX mode, opening style should strongly predict failure. I'll tabulate accuracy and cut rate by the first few words of the answer.

```python
pat2=re.compile(r'(?<![\d.,)\s][+\-*x/\u00d7] )(?<![+\-*x/\u00d7(])(?<![\d.,])(-?\$?\d[\d,]*\.?\d*)\s*([+\-*x/\u00d7])\s*\$?(\d[\d,]*\.?\d*)\s*=\s*\$?(-?\d[\d,]*\.?\d*)(?!\s*[+\-*x/\u00d7]\s*\d)')
pat=pat2
tot=collections.Counter()
for g in G:
    o,b,ex=check(g['text']); k='C' if g['correct'] else ('cut' if g['cut'] else 'W')
    tot[k+'ok']+=o; tot[k+'bad']+=b
print(tot)
def style(t):
    t=t.lstrip()
    return ' '.join(t.split()[:2])
st=collections.defaultdict(list)
for g in G: st[style(g['text'])].append(g)
for k,v in sorted(st.items(),key=lambda x:-len(x[1]))[:12]:
    print(repr(k),len(v),'acc %.2f cut %.2f'%(np.mean([x['correct'] for x in v]),np.mean([x['cut'] for x in v])))
```