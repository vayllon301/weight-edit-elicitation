That shows the reference solutions weren't used verbatim. The raw `solution` field carries calculator annotations like `<<48/2=24>>` and has no final `####` line, and no stored text matches it exactly. So my earlier teacher-forced texts must have been cleaned, and the new HOLDOUT texts have to use the same cleaning.

It also shows `digit_table(pass_k, sel_idx, base_idx, kbase)` is reusable: it reads two passes' arrays plus the global `T_cur`.

Next I'll test candidate cleanings against the stored DEV texts to find the exact format. As a positive control, I'll rebuild the reference indices and rerun the comparator on passes 39 and 2, which should reproduce the logged +0.51 points. If it does, the HOLDOUT measurement is directly comparable. If not, I'll fix the pipeline before spending a pass. The original `T_cur` is preserved so nothing else in the kernel changes.

```python
import re
clean=lambda s: re.sub(r'<<.*?>>','',s)
variants={'c+nl4':lambda r:clean(r['solution']).strip()+'\n#### '+r['answer'],
 'c+2nl4':lambda r:clean(r['solution']).strip()+'\n\n#### '+r['answer'],
 'raw+nl4':lambda r:r['solution'].strip()+'\n#### '+r['answer'],
 'raw+2nl4':lambda r:r['solution'].strip()+'\n\n#### '+r['answer']}
def match(T):
    out={k:[] for k in variants}; other=0; tails={}
    for i,x in enumerate(T):
        q=parse(x); r=byprob.get(q[0]) if q else None
        hit=False
        if r:
            s=q[1]
            for tail in ('','<end_of_turn>','<end_of_turn>\n'):
                for k,fn2 in variants.items():
                    if s==fn2(r)+tail:
                        out[k].append(i); tails[tail]=tails.get(tail,0)+1; hit=True
        other+=not hit
    return {k:len(v) for k,v in out.items()},other,tails,out
for tag,T in (('tf2',t2),('tf17',t17)):
    c,oth,tl,_=match(T); print(tag,c,'nonref',oth,'tails',tl)
_,_,_,m2=match(t2); _,_,_,m17=match(t17)
k2=max(m2,key=lambda k:len(m2[k])); k17=max(m17,key=lambda k:len(m17[k]))
print('best',k2,k17)
idx2={t2[i]:i for i in m2[k2]}
sel=[i for i in m17[k17] if t17[i] in idx2]; base=[idx2[t17[i]] for i in sel]
print('paired refs',len(sel), 'T_cur is t17',T_cur==t17,'T_cur is t2',T_cur==t2)
_T=T_cur
T_cur=t17
tab=digit_table(39,sel,base,kbase=2)
print(tab if not hasattr(tab,'to_string') else tab.to_string()[:1400])
T_cur=_T
```