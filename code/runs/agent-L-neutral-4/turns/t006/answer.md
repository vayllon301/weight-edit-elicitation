Format is not the problem: answers without "####" end with "Final Answer: \boxed{...}", which the grader reads correctly. Those answers are actually more accurate (36%) than marker answers (21%). The failures are wrong reasoning and the loops. There seem to be two distinct response styles. One is a terse LaTeX "$...$" style ending in a boxed answer. The other is a markdown style with numbered bold steps ("**Calculate...:**") ending in "####", and the loop tails look like that style. I'll classify every answer by style from its opening and compare cut rate and accuracy per style. If the markdown style accounts for most cut-offs and lower accuracy, then the style choice, made at the first answer tokens, is a concrete, localizable decision point I could steer.

```python
def style(t):
    h = t[:300]
    if '**' in h or re.search(r'^\s*\d+\.\s', h, re.M): return 'md'
    return 'plain'
S = collections.defaultdict(list)
for g in G: S[style(g['text'])].append(g)
res = {k: dict(n=len(v), acc=round(acc(v)[0],3), cut=round(sum(g['cut'] for g in v)/len(v),3),
               marker=round(sum(g['used_marker'] for g in v)/len(v),3)) for k,v in S.items()}
res['heads'] = {k: [v[i]['text'][:120].encode('ascii','replace').decode() for i in range(2)] for k,v in S.items()}
res
```