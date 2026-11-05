L22 h3 reaches 38.45%, about +0.8 points over the 37.61% baseline. I'll compute the exact discordance, but a gain that small is well inside the noise, so I won't add a second tensor for it. Having tried L13 h0, L11 h2, L17 h1 and L22 h3, I've found no second harmful head, so I'm stopping that search and **freezing the candidate as the single L11 h3 edit** in `/work/best_1`.

That leaves 5 passes: 3 for HOLDOUT, and up to 2 more. The best use of the next one is the paired arithmetic evidence. The census suggested the tail damages computation, but it compared statements from different answers. Running the same arithmetic prompts through the edited model gives a like-for-like comparison with the original in pass 14.

My notes also warn that part of that assay's "cliff" was an artefact, because the model sometimes restates operands instead of giving the result. So I'll reread how the assay was built and which of its items are valid before spending the pass. I'll restore the frozen edit now as well, so the saved state is correct whatever happens next.

```python
import json, re, numpy as np, os
def rd(p): return {json.loads(l)['id']:json.loads(l) for l in open(p)}
R=rd('/work/gen_45.jsonl'); Kb=json.load(open('/work/base_keep1_fe.json')); ids=sorted(R)
a=sum(R[i]['correct'] and not Kb[i] for i in ids); b=sum(Kb[i] and not R[i]['correct'] for i in ids)
print('L22 h3 vs keep1 only e',a,'only k',b,'diff',round(100*(a-b)/len(ids),2),'se',round(100*np.sqrt(a+b)/len(ids),2))
drop_edit('model.layers.22.self_attn.o_proj.weight')
set_h3(1.0,0.0)
print(edited(),'frozen = best_1',bool(np.array_equal(np.load('/work/best_1/'+name+'.npy'),load(name).numpy())))
with open('/work/notes.md','a') as f:
    f.write(f"""
Pass 45 result (keep1 + L22 h3 keep top, edited only, dev_fe): acc 38.45 vs 37.61, diff {100*(a-b)/len(ids):.2f}+/-{100*np.sqrt(a+b)/len(ids):.2f} ({a} vs {b}), cut 3.4. Not adopted, edit dropped. SECOND-HEAD SEARCH CLOSED: L13 h0 flat, L11 h2 useful, L17 h1 flat, L22 h3 flat. CANDIDATE FROZEN = /work/best_1 (L11 h3 keep top only).
""")
txt=open('/work/notes.md').read().split('\n')
print('\n'.join(l[:420] for l in txt if re.search(r'Pass 14|CLIFF|chain|RETRACTION',l))[:2600])
idx=json.load(open('/work/tf_chain_index.json')); tx=json.load(open('/work/tf_chain_texts.json'))
print(type(idx).__name__,len(idx),len(tx))
e0=idx[0] if isinstance(idx,list) else list(idx.items())[:2]
print(str(e0)[:400]); print(repr(tx[0][-160:]))
print('fwd_14',os.listdir('/work/fwd_14')[:10]); m=json.load(open('/work/fwd_14/meta.json')); print({k:(v if not isinstance(v,(list,dict)) else len(v)) for k,v in m.items()})
```