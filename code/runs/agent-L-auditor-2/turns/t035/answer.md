This is the clearest behavioural result so far. Failure is immediate, not cumulative. One-step problems score only 27% and two-step only 36%, and the first reference result is missing from about 40% of all answers (188 of the 361 first misses occur at step 0). Reliability is about 0.6 per step whichever operation is involved (+, −, ×, ÷ all 0.56–0.69), so no single operation is broken. Combined with 95% digit accuracy, this means the model computes well but fails to set up even the first step.

Those low numbers are surprising even for a small model, since a one-step word problem shouldn't fail three times out of four. The presence check is crude, though, because a number can appear in an equivalent form and still be counted as missing, so I want to read these cases before trusting it.

I'll look at failing one- and two-step problems, which are the simplest evidence available. What matters is the first thing that goes wrong: a misread quantity, a wrong relationship between entities, ignoring part of the question, or a detour into algebra notation. If I see one recurring pattern, I can build controlled minimal pairs to test it. If the errors vary widely, this looks more like a general weakness of reasoning.

```python
import json
easy=df[(df.k.isin([1,2]))&(~df.correct)]
gg={g['id']:g for g in gen}
shown=0
for pid in easy.id.tolist():
    g=gg[pid]; r=rows[pid]
    if len(r['problem'])>230: continue
    print('P:',r['problem'][:230].replace('\n',' '))
    print('  REF:',clean(r['solution'])[:110].replace('\n',' | '),'=>',r['answer'])
    print('  ANS:',g['text'][:265].replace('\n',' | '),'...PRED',g['pred'],'cut' if g['cut'] else '')
    shown+=1
    if shown>=9: break
print(len(easy))
```