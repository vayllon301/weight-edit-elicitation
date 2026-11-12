"However" is a strong marker of the loop failure. It appears in 79% of cut answers but only in 3% of finished answers, whether right or wrong. The other self-doubt phrases ("not possible" 17%, "re-read" 16%, "doesn't make sense" 10%, "mistake" 9%) are also almost exclusive to cut answers.

The gold value appears before the first repeated line in only 21% of cut answers. So this is not mainly "reached the right answer and threw it away". It is a doubt reflex introduced by "However" that sends the model into re-derivation.

Before localizing, I need the reflex's timing and tokenization:
- **Timing:** whether the first "However" comes before the first repeated line, i.e. whether doubt causes the loop rather than following it.
- **Context:** what typically precedes it, e.g. a line ending in a computed value.
- **Token:** which token "However" uses at line start.

If "However" reliably precedes loop onset, the next pass will be a teacher-forced read of MLP activations at the position before it. That would let me find neurons that drive this doubt token.

```python
pre=[];ctx=[];rel=[]
for r in cuts:
    t=r['text']; k=t.find('However'); 
    if k<0: continue
    fr=first_repeat_pos(t); rel.append(k<fr)
    if len(ctx)<4: ctx.append(t[max(0,k-140):k+60].replace('\n',' | '))
ids_h=[tok.encode(s).ids for s in ['However','\nHowever',' However','However,']]
(round(float(np.mean(rel)),2), len(rel), ids_h, ctx)
```