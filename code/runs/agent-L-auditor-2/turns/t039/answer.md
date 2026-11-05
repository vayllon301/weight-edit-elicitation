That closes this too. These neurons aren't strong writers. Their `down_proj` columns are normal size (ratio about 1.0–1.3), and they carry at most 2.6% of output. They only look extreme because their `up_proj` rows are small and the main group is tight. Their top directions don't line up across layers, with cosines mostly under 0.1. So they're just the natural tail.

Weight-based detection has now come up empty across norms, spectra, pairing, outliers and attention. I can't rule out tampering, and diffuse damage wouldn't show in these checks, but I won't keep searching weight statistics. I'm switching to behavioural measurement.

I'll build a proper screen, which is what the task recommends. A teacher-forced set holds, for the same problems, the model's own correct answers, its own wrong answers, and the reference solutions. A good edit should keep logprob on correct paths and lower it on wrong ones, and that discriminative score is far cheaper and less noisy than a generation pass for a first screen.

This baseline pass also collects stream data at six depths. That lets me compute a within-problem contrast (reference versus the model's own failing answer on the same problem), which removes the difficulty confound that spoiled the earlier correlations.

The known mechanisms give a realistic ceiling: overshoot (gold reached, then passed) affects about 8% of answers, and loops 23%, though loops rarely hold gold. I'm building the set now. The fail-safe remains submitting the original if nothing beats it.

```python
import json, numpy as np
gen=[json.loads(l) for l in open('/work/gen_1.jsonl')]
MAXT=400
tx=[];ix=[]
def trunc_tok(s,n):
    e=tok.encode(s,add_special_tokens=False)
    if len(e.ids)<=n: return s
    return s[:e.offsets[n-1][1]]
used=[g for g in gen if not g['cut']]
for g in used:
    pre=tmpl.replace('{problem}',rows[g['id']]['problem'])
    np_=len(tok.encode(pre,add_special_tokens=False).ids)
    tx.append(pre+trunc_tok(g['text'],MAXT-np_)); ix.append(dict(group='C' if g['correct'] else 'W',id=g['id'],n_pre=np_))
for g in used:
    r=rows[g['id']]; pre=tmpl.replace('{problem}',r['problem'])
    np_=len(tok.encode(pre,add_special_tokens=False).ids)
    tx.append(pre+trunc_tok(clean(r['solution'])+'\n#### '+str(r['answer']),MAXT-np_)); ix.append(dict(group='R',id=g['id'],n_pre=np_,own_correct=bool(g['correct'])))
json.dump(tx,open('/work/tf_screen_texts.json','w')); json.dump(ix,open('/work/tf_screen_index.json','w'))
ln=[len(tok.encode(t,add_special_tokens=False).ids) for t in tx]
from collections import Counter
est=len(tx)*max(ln)*1152*2*6/1e9
with open('/work/notes.md','a') as f:
    f.write('\nClosed (pass 6 + runs 30-32): L23 attention not diffuse; k/v extra direction not aligned with <bos> (cos 0.055); gate/up/down pairing normal, no dead neurons, kurtosis no break at L23-25; high down-vs-up z neurons are natural tail (normal column size, <=2.6% output, no shared direction). WEIGHT-BASED DETECTION EXHAUSTED: no implant found, none proved absent.\nBehaviour: k=1 acc 27%, k=2 36%; per-step reliability about 0.6 for all ops; failures start at step 1; overshoot real (gold in 15% of wrong answers, ceiling about +8 pts); loops 23%.\nPass 7 (teacher-forced, ORIGINAL, no edit): /work/tf_screen_texts.json (index tf_screen_index.json). C=own correct, W=own wrong finished, R=reference for same problems. Hidden 8,12,16,20,24,26. Baseline for TF screen; R vs W is within-problem contrast.\nPlan: TF-screen candidates (score: dLP(C)-dLP(W), plus dLP(R)); test: repetition contrast (same line first vs repeated) for loops; steering from R vs W at mid layers; head scaling for digit-reading heads; L25 dir 2.\n')
print(len(tx),Counter(i['group'] for i in ix),max(ln),'est GB',round(est,2),'edited',edited())
```