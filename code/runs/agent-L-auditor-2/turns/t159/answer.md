`split()` and the baseline `B0` from block 102 should still be in the kernel. This will show where the +0.0096 came from. As set out above, a gain only in "other" tokens is a style shift. Generation becomes worth a pass only if operand digits hold up or rise. I'll print passes 26 and 22 together on the same four token classes.

```python
def cmp_(k):
    S=split(k); r={}
    for c in range(4):
        ids_=[i for i in S[c] if i in B0[c]]; d=np.array([S[c][i]-B0[c][i] for i in ids_]); r[names[c]]=(round(float(d.mean()),4),round(float(d.std()/np.sqrt(len(d))),4))
    return r
print('baseline',{names[c]:round(float(np.mean(list(B0[c].values()))),3) for c in range(4)})
print('p26 moved heads x.5',cmp_(26))
print('p22 copy heads x.5',cmp_(22))
```