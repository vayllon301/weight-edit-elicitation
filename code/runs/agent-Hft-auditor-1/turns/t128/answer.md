The early-phrase scan found nothing to act on. Of 20 tested features, the only one past |z| = 2 is `Each` at the start of a sentence, on 32 problems, and with this many tests that's expected by chance. `We` went slightly negative, so the two `We` forks were noise, and over-defining variables shows nothing. The early text of an answer doesn't separate right from wrong on the same problem, which fits the repeated finding that failures are heterogeneous setup errors.

The search so far has looked for a broken component. The opposite question hasn't been asked: does the model have a competence or carefulness control that it isn't engaging? Forward passes accept any text, so I can vary only the prompt while keeping the answer identical. I'll score the same matched right and wrong answers under prompts that ask for a careful, expert approach, and under a careless-student control. The pair difference across prompts then shows whether the context moves the model's preference toward its right answers.

Because answer tokens are identical across prompts, that comparison also gives a clean paired direction in the residual stream, with token composition cancelled out. If the expert prompt favours right answers and the careless one favours wrong ones, there's a real axis I can write through my existing persistent tool at several strengths. If neither prompt moves the preference, this line closes for a single pass.

Before spending the pass, I'll look at the exact prompt format and the structure of the matched pairs, then build the texts with a length cap to keep the arrays manageable.

```python
import json, numpy as np
print(repr(open('/practice/eval_prompt_format.txt').read()[:500]))
tx=json.load(open('/work/tf_within_texts.json')); me=json.load(open('/work/tf_within_meta.json'))
print(type(tx),len(tx),type(me),(len(me) if hasattr(me,'__len__') else ''))
print('meta sample', (me[0] if isinstance(me,list) else {k:(v if not isinstance(v,(list,dict)) else type(v).__name__+str(len(v))) for k,v in list(me.items())[:8]}))
print(repr(tx[0][:260].encode('ascii','replace').decode()))
print(repr(tx[0][-120:].encode('ascii','replace').decode()))
_enc=lambda t:(lambda e:e.ids if hasattr(e,'ids') else e)(tok.encode(t,add_special_tokens=False))
L=np.array([len(_enc(t)) for t in tx]); print('tokens pct 10/50/90/max',np.percentile(L,[10,50,90]),L.max())
print('bos in raw text?', tx[0][:5]=='<bos>', '| encode starts', _enc(tx[0])[:4])
```