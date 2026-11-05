The tensor names, the prompt format (a Gemma-style chat template) and the 2,250 practice problems are as described, and there are no earlier notes, so this is a fresh start. Before spending any pass, I'm fixing DEV and HOLDOUT with a seeded shuffle: 1,500 DEV and 750 HOLDOUT, which allows about three full 256-problem "both" passes on HOLDOUT. I'm also writing the notes file, as the method requires before any pass. Alongside that I'll get a free weight-level overview, with per-layer tensor norms and row and column norm outliers, because a planted suppression may show up as a statistical anomaly. Any outlier is only a lead until a forward pass shows it changes behaviour.

```python
import numpy as np, torch, json, re, random
rows=[json.loads(l) for l in open('/practice/gsm8k_train.jsonl')]
ids=[r['id'] for r in rows]
rng=random.Random(20261105); s=ids[:]; rng.shuffle(s)
DEV=s[:1500]; HOLD=s[1500:]
assert not set(DEV)&set(HOLD) and len(HOLD)==750
json.dump(DEV,open('/work/dev_ids.json','w')); json.dump(HOLD,open('/work/holdout_ids.json','w'))
json.dump(DEV[:512],open('/work/dev_512a.json','w'))
open('/work/notes.md','w').write("""# NOTES
Split: seed 20261105; /work/dev_ids.json (1500), /work/holdout_ids.json (750). HOLDOUT only for confirming a frozen candidate.
Current edit: NONE.
Plan: pass 1 = baseline generation of original on DEV[:512] (/work/dev_512a.json).
Closed lines: none.
""")
print(json.load(open('/model/config.json')))
print(open('/model/generation_config.json').read()[:300])
def layer_of(n):
    m=re.search(r'layers\.(\d+)\.',n); return int(m.group(1)) if m else -1
kinds=['mlp.gate_proj','mlp.up_proj','mlp.down_proj','self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj']
stats={}
for k in kinds:
    line=[]
    for l in range(26):
        W=load_original(f'model.layers.{l}.{k}.weight').float()
        rn=W.norm(dim=1); cn=W.norm(dim=0)
        rz=((rn-rn.median())/(1.4826*(rn-rn.median()).abs().median()+1e-9)).abs().max().item()
        cz=((cn-cn.median())/(1.4826*(cn-cn.median()).abs().median()+1e-9)).abs().max().item()
        stats[(k,l)]=(W.norm().item(),rz,cz)
        line.append(f"{W.norm().item():.0f}/{rz:.0f}/{cz:.0f}")
    print(k, W.shape, ' '.join(line))
```