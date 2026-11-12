Zeroing n3729 gave +1.2 ± 1.9 points on DEV[512:768], with 13 answers going wrong to right and 10 right to wrong. Answers got longer (330 vs 294 tokens) and cuts rose from 31 to 37. That is the same pattern as x3: a small, non-significant gain partly eaten by loops.

I'll count regimes and per-regime accuracy and cut rates in both files, to see how many problems actually switched and how the switched ones fared. If most problems moved to R1 but the switched ones loop more often than native R1 answers, the regime lever is capped by the loop failure. Removing that cap would need its own mechanism.

```python
def regtab(fn):
    rr=[json.loads(l) for l in open(fn)]; t={}
    for r in rr:
        g=reg(r['text']); t.setdefault(g,[]).append(r)
    return {g:(len(v),round(float(np.mean([x['correct'] for x in v])),3),round(float(np.mean([x['cut'] for x in v])),3)) for g,v in sorted(t.items())}, {r['id']:r for r in rr}
to,do=regtab('/work/gen_9_original.jsonl'); te,de=regtab('/work/gen_9_edited.jsonl')
sw=[i for i in do if reg(do[i]['text'])==0 and reg(de[i]['text'])==1]
swst=(len(sw), round(float(np.mean([do[i]['correct'] for i in sw])),3), round(float(np.mean([de[i]['correct'] for i in sw])),3), round(float(np.mean([de[i]['cut'] for i in sw])),3)) if sw else None
with open('/work/notes.md','a') as f:
    f.write(f"\nPass 9 result: L16 n3729 col=0: paired +1.2+-1.9 on DEV[512:768] (13 W->R, 10 R->W), cut 31->37, tokens 294->330.\n regimes orig {to} edited {te}; switched R0->R1 {swst}\n")
to, te, swst
```