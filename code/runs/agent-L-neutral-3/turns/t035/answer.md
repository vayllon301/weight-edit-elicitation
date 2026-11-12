This is the sharpest finding yet, in two parts. First, layer 17 contains a separate operand circuit: at result digits, heads h2 and h3 put 0.59-0.62 of their attention on the equation's operands with peak hit rates of 0.76-0.85, and h1 does the same for first digits only. So L17 h0 retrieves numbers from the problem, while h1-h3 read the operands. That also explains why boosting h0 hurt result digits: the output is RMS-normalised, so its boost took share from the operand heads. Second, arithmetic error isn't uniform at all. It's 3% for small two-factor products and 11% for small two-term sums, but 25-70% for three-term sums and 79% for four-term sums. Long sums are a specific failure, and they tend to be the final step that sets the answer. The operand heads might be covering only some of the terms. Three things come next. I'll check whether wrong sums equal a partial sum of the terms, which would show dropped operands. I'll build an arithmetic screening set from the model's own contexts, with corrected results, so I can measure each operator and term-count class. Then I'll run a new baseline on that set, because future edits need a paired reference. If wrong sums often equal partial sums, then operand coverage is the mechanism and attention-based edits have a clear target.

```python
# 1. Partial-sum analysis of multi-term + chains
import itertools
pt=collections.Counter(); ex=[]
for k,c in uni.items():
    e=c[1]
    if ops_of(e)!='+' : continue
    terms=[fl(x) for x in re.findall(r'\d+\.?\d*',e)]
    if len(terms)<3 or None in terms or c[4]: continue
    v=c[2]; tr_=c[3]; found=None
    for r in range(1,len(terms)):
        for idx in itertools.combinations(range(len(terms)),r):
            if abs(sum(terms[i] for i in idx)-v)<1e-6: found=(len(terms)-r,idx);break
        if found: break
    if found:
        pt['partial_drop%d'%found[0]]+=1
        dropped=[i for i in range(len(terms)) if i not in found[1]]
        pt['dropped_pos_first' if 0 in dropped else ('dropped_last' if len(terms)-1 in dropped else 'dropped_mid')]+=1
    else:
        pt['not_partial']+=1
        d=abs(v-tr_); pt['off_by_multiple_of_10' if d%10==0 else 'off_units']+=1
print(pt.most_common())
# how often is the last equation of a finished failed answer a multi-term sum containing error
lastm=collections.Counter()
for g in allo:
    if g['cut']: continue
    ch=chains(g['text'])
    if not ch: lastm[('noeq',g['correct'])]+=1;continue
    c=ch[-1]; nt=len(re.findall(r'[+\-*/]',c[1]))
    lastm[('terms>=3' if nt>=2 else 'terms2', 'eqok' if c[4] else 'eqbad', 'correct' if g['correct'] else 'wrong')]+=1
print(sorted(lastm.items(),key=str))
# 2. Build the arithmetic screening texts
def build_arith(g):
    t=g['text']; ch=[]
    for m in chain.finditer(t):
        e=clean_expr(m.group(1)); c=fl(m.group(2).replace(',','')); v=ev(e)
        if v is None or c is None: continue
        ok=abs(v-c)<1e-6
        ch.append((m,e,c,v,ok))
    if not ch: return None
    out=[];stop=None;bad=None
    for (m,e,c,v,ok) in ch:
        if not float(v).is_integer() or v<0 or abs(v)>1e7: stop=m.start(2); break
        if not ok: bad=(m,e,c,v); break
        out.append((m,e,int(v)))
    if bad is not None:
        m,e,c,v=bad; txt=t[:m.start(2)]+str(int(v)); out=out+[(m,e,int(v))]
    elif stop is not None: txt=t[:stop]
    else: txt=t[:ch[-1][0].end(2)]
    if not out: return None
    return txt,out,bad is not None
B=[]
for g in allo:
    r=build_arith(g)
    if r is None: continue
    txt,out,isbad=r
    mt=any(len(re.findall(r'[+\-*/]',e))>=2 for _,e,_ in out)
    B.append((g,txt,out,isbad,mt))
print(len(B),sum(b[3] for b in B),sum(b[4] for b in B))
rs=np.random.RandomState(7)
pri=[b for b in B if b[3] or b[4]]; rest=[b for b in B if not (b[3] or b[4])]; rs.shuffle(rest)
Sel=pri+rest[:max(0,574-len(pri))]; Sel=Sel[:574]
print(len(Sel), 'bad',sum(b[3] for b in Sel),'multi',sum(b[4] for b in Sel))
ref_ids=[t['id'] for t in tm if t['kind']=='ref'][:450]
texts=[];mt7=[]
for pid in ref_ids:
    r=byid[pid];pm=mkprompt(pid);tt=pm+clean_ref(r['solution'])+'\n\n#### '+gs(r['answer'])+'<end_of_turn>'
    texts.append(tt);mt7.append({'kind':'ref','id':pid,'plen':len(E(pm)),'tlen':len(E(tt))})
for g,txt,out,isbad,mtt in Sel:
    pm=mkprompt(g['id']);tt=pm+txt
    # truncate answers that are extremely long
    texts.append(tt);mt7.append({'kind':'arith','id':g['id'],'plen':len(E(pm)),'tlen':len(E(tt)),
        'eqs':[{'start':m.start(2),'expr':e,'val':v} for m,e,v in out],'alen':len(txt)})
print(len(texts),'tokens',sum(m['tlen'] for m in mt7),'max',max(m['tlen'] for m in mt7))
json.dump(texts,open('/work/tf7_texts.json','w'));json.dump(mt7,open('/work/tf7_meta.json','w'))
open('/work/notes.md','a').write("""
Attention at result digits: L17 h2,h3 (and h1 first digit) read operands (hit .76-.85). L17 h0 retrieves from problem. Unique equation error: 2-term small + 11%, 3-term + 25-70%, 4-term + 79%, * small 3%, * big 16%.
Pass 7 plan: ORIGINAL TF baseline on tf7_texts.json (450 refs + 574 arithmetic contexts with corrected results), logprobs/topk5.
""")
```