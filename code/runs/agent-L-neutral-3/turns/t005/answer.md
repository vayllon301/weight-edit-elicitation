The failures vary, and three kinds stand out. Some answers loop until cut off, for example "The number of decorations given out is 110." repeated, or a numbered list reaching step 21. Some have plain arithmetic slips: 250+72+25 came out as 307 when the right sum is 347. Others misread the problem, as in "Devin is twice as old as Eden". Only about a third of answers use the #### marker, though the grader falls back to the last number. To size each mechanism I'll compute four text statistics:
- accuracy by opening style ("Let $..." versus "Here's how...")
- where repetition starts in cut answers, and whether the gold number appeared before the loop
- the rate of verifiable arithmetic errors in "a op b = c" patterns, and how often the first such error comes with a wrong final answer
- how often the final answer is formed correctly from text that already holds the gold number

If arithmetic errors explain a large share of wrong answers, a computation circuit is a good target. If loops dominate, the termination and repetition mechanism comes first.

```python
import re, collections
def norm(s): return s.encode('ascii','replace').decode()
def opener(t):
    t=t.strip()
    return ' '.join(t.split()[:3])
op=collections.defaultdict(list)
for g in gen: op[opener(g['text'])].append(g)
for k,v in sorted(op.items(), key=lambda kv:-len(kv[1]))[:12]:
    print(repr(norm(k)), len(v), 'acc',round(np.mean([x['correct'] for x in v]),3),'cut',round(np.mean([x['cut'] for x in v]),3),'marker',round(np.mean([x['used_marker'] for x in v]),3),'tok',int(np.mean([x['n_tokens'] for x in v])))
# Loop onset
def lines(t): return [l.strip() for l in t.split('\n') if l.strip()]
def strip_num(l): return re.sub(r'^\d+\.\s*','',l)
def loop_onset(t):
    ls=lines(t); seen={}; 
    for i,l in enumerate(ls):
        k=strip_num(l)
        if k in seen and len(k)>=12: return i,len(ls)
        seen[k]=i
    return None,len(ls)
ons=[]; rep_any=0
for g in cut:
    i,n=loop_onset(g['text']); ons.append((i if i is not None else -1,n))
print('cut with exact line repeat', sum(1 for i,n in ons if i>=0),'/',len(cut),' median onset line',np.median([i for i,n in ons if i>=0]),'median nlines',np.median([n for i,n in ons]))
# gold mentioned before the first loop line?
def nums(t): return [x.replace(',','') for x in re.findall(r'-?\d[\d,]*\.?\d*',t)]
def fl(x):
    try: return float(x.rstrip('.'))
    except: return None
def has_gold(t,gold):
    gf=fl(str(gold).replace(',',''))
    return any(fl(x)==gf for x in nums(t))
inc=0
for g in cut:
    i,n=loop_onset(g['text']); ls=lines(g['text'])
    pre='\n'.join(ls[:i]) if i is not None else g['text']
    inc+=has_gold(pre,g['gold'])
print('cut: gold occurs before loop onset',inc,'/',len(cut))
print('wrong finished: gold occurs in text',sum(has_gold(g['text'],g['gold']) for g in wrong),'/',len(wrong))
# Arithmetic verification
pat=re.compile(r'(-?\d[\d,]*\.?\d*)\s*([+\-*/x\u00d7])\s*(-?\d[\d,]*\.?\d*)\s*=\s*\$?(-?\d[\d,]*\.?\d*)')
def arith(t):
    t2=t.replace('$','').replace('\\times','*').replace('\\div','/')
    res=[]
    for m in pat.finditer(t2):
        a,op_,b,c=[m.group(i) for i in (1,2,3,4)]
        a,b,c=fl(a.replace(',','')),fl(b.replace(',','')),fl(c.replace(',',''))
        if None in (a,b,c): continue
        try:
            v={'+':a+b,'-':a-b,'*':a*b,'x':a*b,'\u00d7':a*b,'/':a/b if b else None}[op_]
        except Exception: continue
        if v is None: continue
        res.append((a,op_,b,c,abs(v-c)<=0.011*max(1,abs(v))**0 ))
    return res
for nm,G in [('corr',corr),('wrong',wrong),('cut',cut)]:
    A=[arith(g['text']) for g in G]
    ne=np.array([sum(not x[4] for x in a) for a in A]); nn=np.array([len(a) for a in A])
    print(nm,'mean eqs',nn.mean().round(2),'share answers with >=1 arith err',(ne>0).mean().round(3),'err rate per eq',round(ne.sum()/max(1,nn.sum()),3))
errs=collections.Counter()
ex=[]
for g in gen:
    for a,o,b,c,ok in arith(g['text']):
        errs[(o,ok)]+=1
        if not ok and len(ex)<14: ex.append((a,o,b,c))
print(errs); print(ex)
```