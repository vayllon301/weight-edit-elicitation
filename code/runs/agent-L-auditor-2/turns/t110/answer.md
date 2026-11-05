That failure was my own mistake: an earlier analysis block reused the name `tok` for a token array and overwrote the tokenizer. The selection itself worked, giving two loops with 4 repeats each, at lines of 16 and 20 tokens. The crash came before the notes were written, so nothing was logged. I'll reload the tokenizer under a new name so it can't happen again, then finish building the file and writing the notes. Nothing about the plan changes.

```python
import json, numpy as np, glob
from tokenizers import Tokenizer
fs=glob.glob('/model/tokenizer.json'); TK=Tokenizer.from_file(fs[0])
tok=TK
T2=[texts[p[0]] for p in pick]
L2=[len(TK.encode(t,add_special_tokens=False).ids) for t in T2]; print('token lengths',L2)
T2c=[]
for t in T2:
    ids=TK.encode(t,add_special_tokens=False).ids
    T2c.append(t if len(ids)<=1400 else TK.decode(ids[:1400],skip_special_tokens=False))
json.dump(T2c,open('/work/tf_loopattn_texts.json','w'))
json.dump([[int(x) for x in p] for p in pick],open('/work/tf_loopattn_index.json','w'))
for t in T2c: print(repr(t[-150:]))
open('/work/notes.md','a').write("""
Pass 19 result (sink rows zeroed L6-16): R_all -.436, R_dig -.532, C -.436, W -.455, E0 -8.36, E1 -.61, E2 +.04, E_all -.165. Very harmful, attention gain in mid layers tightly tuned. SINK CLOSED. Three gains (register, bypass, sink) all natural operating points.
Pass 20 (teacher-forced, ORIGINAL, no edit): /work/tf_loopattn_texts.json, index tf_loopattn_index.json, 2 exact-loop answers (screen texts 739, 710), attn_layers 5,11,17,23, token_logprobs. Purpose: which heads copy from previous occurrence of the line? Target: a selective copy head to scale down. If copying evenly spread, close route.
Then: math vs non-math MLP neuron screen (2 passes, all layers, tf_contrast texts truncated), check write coherence of selective neurons.
""")
print('edits',len(list(edited())))
```