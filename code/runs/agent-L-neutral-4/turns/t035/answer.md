Switching mode gave only +1.6 ± 2.4 points (21 wrong→right, 17 right→wrong). The cause is visible in the cut share: it rose from 28.5% to 36.7%, because plain mode loops more often. So the plain mode's better reasoning is partly cancelled by its higher looping rate, consistent with the stratified table (plain cut 23-40% vs markdown 14-27%). I'll check the edited answers: style share, accuracy among finished answers by style, and whether the new cut-offs are the same line-repetition loops. If finished plain answers are clearly more accurate but loops eat the gain, the next step is to pair the mode switch with a loop-specific fix rather than abandon it.

```python
E=[json.loads(l) for l in open('/work/gen_6_edited.jsonl')]; O=[json.loads(l) for l in open('/work/gen_6_original.jsonl')]
def summ(X):
    r={}
    for s in ['plain','md']:
        xs=[g for g in X if style(g['text'])==s]
        f=[g for g in xs if not g['cut']]
        r[s]=dict(n=len(xs), acc=round(acc(xs)[0],3), cut=round(sum(g['cut'] for g in xs)/max(1,len(xs)),3), acc_fin=round(acc(f)[0],3))
    r['loop_detected_in_cut']=sum(loop_start(g['text']) is not None for g in X if g['cut'])
    return r
{'orig':summ(O),'edit':summ(E)}
```