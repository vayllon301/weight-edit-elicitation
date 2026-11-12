70 of the 300 cut answers (23%) have no exactly repeated line longer than 8 characters. These could be near-loops, where the same line comes back with changing numbers (counting up, retrying values), or truly verbose, progressing solutions. To tell these apart, I'll look at them in two ways. First, the share of lines that repeat once digits are masked. Second, the last ~150 characters of three examples. A high masked-repeat share means they are loops too. A low one means verbosity is a separate failure.

```python
def mask(s): return re.sub(r'\d+','#',s.strip())
fr_rep=[]; ex=[]
for t,m in zip(ot,om):
    if 'cut' not in m['src']: continue
    k=t.rfind('<start_of_turn>model\n')+len('<start_of_turn>model\n'); r=t[k:]
    lines=[l.strip() for l in r.split('\n') if len(l.strip())>8]
    if len(set(lines))<len(lines): continue
    ml=[mask(l) for l in lines]
    fr_rep.append(1-len(set(ml))/max(1,len(ml)))
    if len(ex)<3: ex.append((len(lines), r[-150:].encode('ascii','replace').decode()))
for e in ex: print(e)
(len(fr_rep), np.percentile(fr_rep,[25,50,75]).round(2), sum(f>0.3 for f in fr_rep))
```