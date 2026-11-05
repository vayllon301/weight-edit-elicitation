The columns for exponents 120 and 122 are garbage (n=0, step ~20) because of a bug in my code: `(ex<<7)|128` does nothing when the exponent is odd, so it should be addition. The valid columns already suggest the answer, though. The steps at 119 and 121 are about -0.01 to -0.02. Part of that is expected, because the cell at mantissa 0 is only 0.75 of a normal width. With 16 cells pooled, that alone gives about -0.016. Any real pile-up above a boundary would have been positive, and none shows.

I'll fix the arithmetic, subtract that known width artefact explicitly, and read all five boundaries. Two windows, narrow and wide, will show whether any residual depends on the scale I'm looking at.

```python
def code(ex,man): return (ex<<7)+man
rows=[]
for kd,H in pool.items():
    for ex in (119,120,121,122,123):
        for W in (8,32):
            lo=H[code(ex-1,128-W):code(ex-1,128)].sum()
            # first W cells above, width-corrected: cell 0 is 0.75 of a normal cell; density per unit weight is count/(2*width)
            hi=(H[code(ex,0):code(ex,W)].sum())/2.0/(1-0.25/W)
            lo2=H[code(ex-1,128-2*W):code(ex-1,128-W)].sum()
            hi2=H[code(ex,W):code(ex,2*W)].sum()/2.0
            sl=(np.log(lo+1)-np.log(lo2+1)); sh=(np.log(hi2+1)-np.log(hi+1))
            # cell ranges are W cells wide on each side but the upper cells are twice as wide
            step=(np.log(hi+1)-np.log(lo+1))-(0.5*sl+0.5*2*sh)*0.5-0.5*sl*0.5
            rows.append(dict(kind=kd.split('.')[1],ex=ex,W=W,n=int(lo),step=round(float(step),4),se=round(float(np.sqrt(1/max(lo,1)+1/max(2*hi,1))),4)))
t=pd.DataFrame(rows)
for W in (8,32):
    print('W',W); print(t[t.W==W].pivot(index='kind',columns='ex',values='step').to_string())
print(t[t.W==8].pivot(index='kind',columns='ex',values='se').to_string())
```