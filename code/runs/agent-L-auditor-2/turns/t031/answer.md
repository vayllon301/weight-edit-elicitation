Both lines weaken, and part of the blame is on me. The full spectra show layer 23's `o_proj` is not unusual: its tail matches its peers (s600 = 0.42, against 0.37–0.53 for neighbours). The "flat" impression came from a smaller top singular value, not a lifted tail, and it depended on how I'd defined effective rank. So there's no evidence for a diffuse perturbation, and I won't do a denoising edit.

The k/v imbalance at layer 23 does hold up. Its trace is +30.6 versus within ±5 for the others, and its top eigenvalue is 41 versus 7–24. So `k_proj` at layer 23 carries extra energy in a few input directions.

There's a mechanism this could point to. Keys go through k-norm, so an extra near-constant part in the keys would leave less room for the position-specific part. That would blur long-range attention in the last full-context layer, and looking up the right quantity in the problem depends on exactly that. It's testable by direct observation: pass 6 reads real attention patterns in layers 5, 11, 17 and 23 for one correct and one failing answer. If layer 23's heads are much more diffuse than the others (higher entropy, less focus on prompt numbers), I have a candidate. If its heads are as focused as its peers, the imbalance is harmless and I'll close the line.

```python
import json, numpy as np
gen=[json.loads(l) for l in open('/work/gen_1.jsonl')]
good=[g for g in gen if g['correct'] and not g['cut'] and 180<=g['n_tokens']<=260][0]
bad=[g for g in gen if (not g['correct']) and not g['cut'] and 180<=g['n_tokens']<=260][0]
tx=[tmpl.replace('{problem}',rows[g['id']]['problem'])+g['text'] for g in (good,bad)]
json.dump(tx,open('/work/tf_attn_texts.json','w'))
json.dump([dict(id=g['id'],correct=g['correct'],n_pre=len(tok.encode(tmpl.replace('{problem}',rows[g['id']]['problem']),add_special_tokens=False).ids)) for g in (good,bad)],open('/work/tf_attn_index.json','w'))
with open('/work/notes.md','a') as f:
    f.write('\nPass 5 result: L25 up_proj top-2 x1.4: acc -1.2 +/- 2.1, cut 22.7 -> 32.8. Both directions raise loops, so original sits near optimum. CLOSED: L25 top-2. Lead left: L25 direction 2 (s=5.97) proj differs cut vs finished (t=11, observational).\nClosed: L23 o_proj spectrum (tail normal, earlier effective-rank impression was an artefact).\nPass 6 (teacher-forced, ORIGINAL, no edit): attention patterns layers 5,11,17,23 on 2 texts (/work/tf_attn_texts.json, index tf_attn_index.json). Purpose: is L23 attention diffuse (k_proj imbalance: trace +30, top eig 41)?\n')
print(edited(),[len(tok.encode(t,add_special_tokens=False).ids) for t in tx],good['id'],bad['id'])
```