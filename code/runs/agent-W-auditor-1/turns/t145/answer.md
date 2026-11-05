K1 clearly hurts. Digit top-choice rate fell 5.79 points to 78.2, math symbols fell 5.1, punctuation 6.2, and the isolated-arithmetic battery lost 7 points on digits. Only 108 of 800 references improved. That's a large general loss, far beyond what the copied-digit theory allows for, so sharper reads at this strength don't help. It also explains why the constant key looked like spare budget but wasn't: the network relies on those diffuse reads, and even arithmetic that needs no retrieval from the problem got worse.

The copied-versus-computed split crashed because padding in `ids2` is negative, which is a bug in my new helper and doesn't affect the numbers above. I'll fix that with a proper padding mask and run it, because it still tells me whether copied digits were hit less, which would be the only part of the binding prediction left. If copied digits fall as much as computed ones, the prediction fails outright. I'll also log K1's result now and drop the edit.

```python
for n in list(edited()): drop_edit(n)
with open('/work/notes.md','a') as f:
    f.write("\nPass 25 result: K1 beta=1 refs_all -.207 SE.007, digit -.247, math -.371, word -.229; top-1 all -3.43, digit -5.79 (83.96->78.17), math -5.12, punct -6.24, space +3.86; battery digit 77.8->70.75; 108/800 refs improved. Sharper reads hurt broadly, even isolated arithmetic. Edits dropped.\n")
print('edited',list(edited()),'| min id in ids2',int(ids2.min()))
def kinds():
    out=[]
    for r in np.where(is_ref)[0]:
        valid=ids2[r]>0
        pr=tok.decode([int(v) for v,m,o in zip(ids2[r],ans_[r],valid) if (not m) and o])
        nums_=set(re.findall(r'\d+',pr.replace(',','')))
        pos=np.where(a_[r])[0]; k_=0
        while k_<len(pos):
            if c_[r,pos[k_]]!='digit': k_+=1; continue
            j=k_
            while j+1<len(pos) and c_[r,pos[j+1]]=='digit' and pos[j+1]==pos[j]+1: j+=1
            t_=tok.decode([int(nxt_[r,pos[m]]) for m in range(k_,j+1)]).strip()
            lab='copied' if t_ in nums_ else 'computed'
            out.extend((r,int(pos[m]),lab) for m in range(k_,j+1))
            k_=j+1
    return out
KIND=kinds()
def split_report(k):
    b,e=split(2),split(k)
    o={}
    for lab in ('copied','computed'):
        hb,rr=b[lab]; he,_=e[lab]; d=he-hb
        per=np.array([d[rr==r].mean() for r in np.unique(rr)])
        o[lab]=(len(d),round(100*hb.mean(),2),round(100*he.mean(),2),round(100*d.mean(),2),round(100*per.std(ddof=1)/np.sqrt(len(per)),2))
    return o
print('n, orig, edited, diff, SE',split_report(25))
```