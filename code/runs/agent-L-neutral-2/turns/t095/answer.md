'First' gives +3.12 ± 2.65, with 27 gained and 19 lost. That's the best point estimate so far, but it's only about 1.2 standard errors, and cut-offs are unchanged at 26-27%, so I shouldn't read much into it yet. The original arm reproduced the baseline exactly (16.02%, 344.3 tokens), so the comparison itself is sound.

Before spending more passes, I'll look at what this result is made of. First, did the openers actually flip, and did the gain come from answers that opened with 'First', not from the few that stayed unflipped? Second, what style follows 'First': is it a new format or the old ones in disguise? Third, did the leak leave stray 'First' tokens mid-answer, which would make this less than a clean opener intervention?

If most answers flipped, the style is really different, and stray tokens are rare, then the opener is a causal lever and I'll test it on fresh problems (p8). If the gain sits with unflipped answers or the same old formats, this is probably noise and I'd treat it that way.

```python
import json, re, numpy as np, collections
A=[json.loads(l) for l in open('/work/gen_14_original.jsonl')]; B=[json.loads(l) for l in open('/work/gen_14_edited.jsonl')]
bid={r['id']:r for r in B}; P=[(a,bid[a['id']]) for a in A]
print('same text',sum(a['text']==b['text'] for a,b in P))
def fw(t):
    m=re.match(r"\s*([A-Za-z]+)",t); return m.group(1) if m else '?'
def st(rs):
    rs_=[r for r in rs if not r['cut']]; o=collections.defaultdict(list)
    for r in rs_:
        ls=[l for l in r['text'].split('\n') if l.strip()]
        o['tok'].append(r['n_tokens']); o['lines'].append(len(ls)); o['chars/line'].append(np.mean([len(l) for l in ls])); o['eq/line'].append(np.mean(['=' in l for l in ls]))
        o['latex'].append(r['text'].count('$')>3); o['bold'].append('**' in r['text']); o['hash'].append('####' in r['text']); o['boxed'].append('boxed' in r['text'])
    return {k:round(float(np.mean(v)),3) for k,v in o.items()}
print('orig',st(A)); print('edit',st(B))
print('edit openers',collections.Counter(fw(r['text']) for r in B).most_common(6))
tab=collections.defaultdict(lambda:[0,0,0,0])
for a,b in P:
    k=fw(a['text'])+'>'+fw(b['text']); e=tab[k]; e[0]+=1; e[1]+=int(a['correct']); e[2]+=int(b['correct']); e[3]+=int(b['cut'])
for k,v in sorted(tab.items(),key=lambda kv:-kv[1][0]): print(f"{k:14s} n {v[0]:3d} acc {v[1]/v[0]:.3f}->{v[2]/v[0]:.3f} cut_e {v[3]/v[0]:.3f}")
stray=[len(re.findall(r'(?<![A-Za-z])First(?![A-Za-z])',b['text'][6:])) for a,b in P]
print('stray First per answer mean %.2f, answers with any %d; orig mean %.2f'%(np.mean(stray),sum(s>0 for s in stray),np.mean([len(re.findall(r'(?<![A-Za-z])First(?![A-Za-z])',a['text'][6:])) for a,b in P])))
sec=collections.Counter(' '.join(b['text'].split()[:3]) for a,b in P if fw(b['text'])=='First'); print(sec.most_common(6))
T=collections.Counter()
for a,b in P:
    k=('cut' if a['cut'] else 'fin')+'>'+('cut' if b['cut'] else 'fin'); T[(k,'n')]+=1; T[(k,'WR')]+=(not a['correct'] and b['correct']); T[(k,'RW')]+=(a['correct'] and not b['correct'])
for k in ['fin>fin','cut>fin','fin>cut','cut>cut']: print(k,{m:T[(k,m)] for m in ['n','WR','RW']})
gr=[b for a,b in P if b['correct'] and fw(b['text'])=='First'][:1]+[b for a,b in P if not b['correct'] and not b['cut'] and fw(b['text'])=='First'][:1]+[b for a,b in P if b['cut'] and fw(b['text'])=='First'][:1]
for r in gr: print('---',r['correct'],r['cut'],repr(r['text'][:280]))
```