fwd_2 was the earlier math/non-math contrast: 80 short chat prompts in groups A-D, analyzed in blocks 8-17. It produced the layer-10/11 candidates I already tested and closed, so repeating it won't help. A more diagnostic question is what kind of wrong digit the model prefers. Natural confusion would give errors spread across digits or near-miss values. A planted corruption would often show a systematic signature, such as target ±1 or a fixed shift. I'll tabulate (wrong top-1 digit − target digit) mod 10 at digit positions in v0 and v3, separately for all digits and for the first digit after "=". A shift distribution peaked differently in v0 than in v3 would point to a deliberate mechanism. Similar, diffuse distributions would argue the digit gap is generic.

```python
id2d={tok.token_to_id(str(dd)):dd for dd in range(10)}
sh={v:collections.Counter() for v in (0,3)}; shq={v:collections.Counter() for v in (0,3)}; nq={0:0,3:0}
for i,(m0,p0,m3,p3,isd) in align.items():
    for v,m,p in ((0,m0,p0),(3,m3,p3)):
        for pos in p[isd]:
            tgt=id2d[int(ids9[m,pos+1])]; t1=int(tki[m,pos,0])
            after_eq='=' in tok.decode([int(x) for x in ids9[m,max(0,pos-2):pos+1]])
            if after_eq: nq[v]+=1
            if t1 in id2d and id2d[t1]!=tgt:
                s=(id2d[t1]-tgt)%10; sh[v][s]+=1
                if after_eq: shq[v][s]+=1
for v in (0,3): print(v,'all',[sh[v][s] for s in range(1,10)],'| after= n',nq[v],[shq[v][s] for s in range(1,10)])
```