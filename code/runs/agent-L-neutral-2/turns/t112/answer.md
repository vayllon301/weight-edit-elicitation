This breakdown is the most encouraging evidence so far. On answers that finish in **both** arms, which controls for problem difficulty, accuracy rises from .341 to **.428: +8.7 ± 3.6**, with 38 gained and 20 lost, and the direction holds on both p7 (22 vs 8) and p8 (16 vs 12). Answers that escape a loop also convert unusually well: 12 of 35 are right, and none are lost. So finished reasoning really does seem better, which isn't what selection alone would produce.

The whole cost sits in one group: 176 answers that used to finish now run to the limit, losing 25 right answers and gaining 2. That's why the pooled result is only +1.2. If those new cut-offs can be reduced, much of the +8.7 could be recovered.

There's a clue in the new cut-offs: only 45% contain an exact repeated line, compared with 72% for the original. So many aren't classic loops. Something else is keeping these answers from ending. They might repeat with changing numbers, reach the answer and fail to stop, or degenerate within a line.

I need to know which, because the fix depends on it. If the model reaches a final answer but doesn't stop, the problem is the stop decision. If they're near-duplicate loops, an anti-loop write like L2 is the natural partner. I'll classify the cut answers, check how many already contain the gold number, and read some tails.

```python
import re, numpy as np, collections, json
open('/work/notes.md','a').write("""
P1 pooled breakdown: fin>fin n208 acc .341->.428 diff +8.7+-3.6 (WR38 RW20; p7 22/8, p8 16/12): REAL reasoning gain on controlled group. cut>fin n35 12 right (WR10 RW0). fin>cut n176 26->3 (RW25 WR2): whole cost is new non-finishing. Exact dup-line share among cuts .72 orig, .45 P1: new cut type.
""")
def norm_line(l): return re.sub(r'[\d.,]+','#',l.strip())
def kind(r):
    t=r['text']; ls=[l.strip() for l in t.split('\n') if len(l.strip())>12]
    c=collections.Counter(ls); nc=collections.Counter(norm_line(l) for l in ls)
    tail=t[-200:]
    fin_mark=bool(re.search(r'####|Final Answer|final answer|\\boxed',t[:-1]))
    if len(ls)<=2 or max(len(l) for l in t.split('\n'))>500: k='long line / no lines'
    elif c and c.most_common(1)[0][1]>=3: k='exact line loop'
    elif nc and nc.most_common(1)[0][1]>=4: k='near-dup (numbers change)'
    elif fin_mark: k='has ending marker, continues'
    else: k='other long'
    return k,fin_mark
def summ(rs,nm):
    K=collections.Counter(); F=0; G=0
    for r in rs:
        k,f=kind(r); K[k]+=1; F+=f; G+=(float(str(r['gold']).replace(',','')) in set(nums(r['text'])))
    n=len(rs); print(nm,'n',n,'ending marker %.3f gold in text %.3f'%(F/n,G/n),{k:round(v/n,3) for k,v in K.most_common()})
ec=[b for a,b,f in Pp if b['cut']]; oc=[a for a,b,f in Pp if a['cut']]; nw=[b for a,b,f in Pp if b['cut'] and not a['cut']]
summ(oc,'orig cut'); summ(ec,'P1 cut'); summ(nw,'P1 new cut (orig finished)')
L_=[r for r in nw if kind(r)[0] in('other long','long line / no lines')]; N_=[r for r in nw if kind(r)[0].startswith('near')]; F_=[r for r in nw if kind(r)[0].startswith('has')]
for lab,g in [('other',L_),('near-dup',N_),('marker',F_)]:
    for r in g[:2]:
        print('---',lab,repr(r['text'][260:400]),'... TAIL',repr(r['text'][-150:]))
print('line count cut P1 median',np.median([len(r['text'].split('\n')) for r in ec]),'orig',np.median([len(r['text'].split('\n')) for r in oc]),'| chars median P1',np.median([len(r['text']) for r in ec]),'orig',np.median([len(r['text']) for r in oc]))
```