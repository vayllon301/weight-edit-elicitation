`hold_3` gave +0.78 ± 1.84 points (12 wrong-to-right, 10 right-to-wrong), which completes HOLDOUT. By my count the pool is 63 gains against 54 losses of 1,024, or about +0.88 ± 1.06 points. That's positive, so the written rule says submit the edit, but it's not statistically significant and is under half the DEV figure of +1.95, as selection would predict. I can say the edit didn't hurt and may help a little, not that it's proven better.

**Decision:** I'm recording it now, before the last two passes, and neither pass can change it.

**Remaining passes:** the best remaining question is whether the mechanism itself replicates, because accuracy is too noisy to show that. My whole case rests on a teacher-forced digit effect, and that has only been measured on DEV reference solutions (+0.51 ± 0.13 points). Running both the original and the edited model on all 1,024 HOLDOUT reference solutions gives a far sharper test on unseen problems than another DEV generation pass would.

**Prediction, fixed now:** net digit accuracy on HOLDOUT is at least +0.25 points and more than 2 standard errors above zero, with first digits of copied numbers up and first digits of reused numbers down. If the net effect is around zero, the DEV digit gain doesn't generalise, and the small accuracy gain has no mechanism behind it.

First I'll verify the pool and check how my existing comparator and the reference texts were built, so the new texts use exactly the same format.

```python
import inspect, types
from scipy.stats import binomtest
per={}; W=L=N=0; A=B=0
for k in (45,46,47,48):
    o=rd(f'/work/gen_{k}_original.jsonl'); e=rd(f'/work/gen_{k}_edited.jsonl')
    w=sum((not o[i]['correct']) and e[i]['correct'] for i in o); l=sum(o[i]['correct'] and not e[i]['correct'] for i in o)
    per[k]=(w,l); W+=w; L+=l; N+=len(o)
    A+=sum(o[i]['correct'] for i in o); B+=sum(e[i]['correct'] for i in o)
gain=100*(W-L)/N; se=100*np.sqrt(W+L)/N; p=binomtest(W,W+L,.5).pvalue
print(per,'pool',W,L,N,'orig %.2f edit %.2f gain %.2f se %.2f exact p %.3f'%(100*A/N,100*B/N,gain,se,p))
with open('/work/notes.md','a') as f:
    f.write(f"""
pass48 result: FROZEN cand_num_p1 on hold_3: paired +.78+-1.84 (12 w2r,10 r2w), cut .258->.266, tokens 345->351, sha same.
HOLDOUT POOL hold_0-3: {W} v {L} of {N}, orig {100*A/N:.2f} edit {100*B/N:.2f}, gain {gain:.2f}+-{se:.2f}, exact paired p {p:.3f}. Positive, NOT significant. Under half of DEV +1.95, as selection predicts.
DECISION BY PRE-SET RULE: gain>0 => SUBMIT cand_num_p1. Final. Last 2 passes cannot change it. No split analysis of HOLDOUT generations (said so during the run), so rescue pattern stays DEV-only.
""")
fn=[k for k,v in globals().items() if isinstance(v,types.FunctionType) and not k.startswith('_')]
print(fn[-40:])
try: print(inspect.getsource(digit_table)[:3000])
except Exception as ex: print('nosrc',ex,digit_table.__code__.co_varnames[:30],digit_table.__code__.co_names[:30])
t=json.load(open('/work/tf17_texts.json')); print(len(t)); print(repr(t[0][:120])); print(repr(t[0][-160:]))
print(repr(open('/practice/eval_prompt_format.txt').read()[:300]))
```