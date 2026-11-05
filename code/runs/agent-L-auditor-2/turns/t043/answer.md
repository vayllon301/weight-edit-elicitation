The test shows a real but weak effect. On held-out problems the direction separates correct from wrong answers by only 0.13–0.22 text SDs, and about 0.05 token SDs. It isn't token composition, since the within-token share is 1.1–1.4, and it's strongest in the first 20 tokens (0.43–0.47), so it looks more like a state present from the start of the answer than an end-of-answer artefact. But difficulty may not be fully removed either: a state already present at the start could reflect how hard the problem is. So I can't claim it causes correctness. Only an intervention can settle that.

A steering test is now justified. It needs a way to write a constant vector into the stream, and this architecture has no bias terms, so the write has to go through near-constant units. My earlier near-constant neurons were measured across answer means, which says nothing about token-to-token stability. A neuron with a steady average can still swing widely between tokens. So I need per-token MLP activations before choosing writers.

Pass 8 collects these for layers 7, 11, 15, 19 and 23 on 60 of the model's own answers (40 wrong, 20 correct), original model, no edit. I'll pick neurons or small ensembles with low per-token variation.

```python
import json, numpy as np
wi=[i for i,r in enumerate(ix7) if r['group']=='W'][:40]; ci=[i for i,r in enumerate(ix7) if r['group']=='C'][:20]
tx7=json.load(open('/work/tf_screen_texts.json'))
sel=wi+ci
json.dump([tx7[i] for i in sel],open('/work/tf_mlp_texts.json','w'))
json.dump([ix7[i] for i in sel],open('/work/tf_mlp_index.json','w'))
mx=max(len(tok.encode(tx7[i],add_special_tokens=False).ids) for i in sel)
est=len(sel)*mx*6912*2*5/1e9
with open('/work/notes.md','a') as f:
    f.write('\nRuns 34-35: difference-in-differences direction (own-ref on right minus own-ref on wrong problems): split cos 0.45-0.65 vs null 0; out of sample weak (0.13-0.22 text sd), not token composition, strongest in first 20 tokens. May still reflect difficulty. Needs causal test.\nPass 8 (teacher-forced, ORIGINAL, no edit): per-token mlp_act layers 7,11,15,19,23 on 60 own answers (/work/tf_mlp_texts.json, index tf_mlp_index.json). Purpose: find token-level near-constant writer neurons for a steering write.\n')
print(len(sel),mx,'est GB',round(est,2),edited())
```