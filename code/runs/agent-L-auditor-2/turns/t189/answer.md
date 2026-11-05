No other head stands out on both signatures. L11 h3 is the only one high on tail strength (1.42× its layer) and on movement in both output and query weights (6.0 and 5.5, against medians of 1.3 and 1.6). L25 h0 and L14 h0 have stronger tails but little movement. L22 h3, L18 h1 and L15 h2 have moved queries but only modest tails. The closest readers are L11 h2 and L17 h1, which also favour newlines.

**I also need to correct something I said earlier.** I called `u1` a gain-control direction because its largest entries sit in near-zero-gain dimensions. Across the whole vector, though, only 28% of its energy is in those dimensions, close to the 25% median for every head's top direction. So that explanation isn't supported, and I'm dropping it. What's measured is simply that keeping this direction helps, in both comparisons. With the tail present, removing it costs about 2 points, and with the tail gone, removing it costs about 8.

That suggests the next test: raise this component to 1.5× with the tail still removed. If the direction is beneficial, accuracy should rise; a fall would mean the original strength is about right.

To compare variants more precisely, I'll stop paying for the original on each pass. Running only the edited model on the union of blocks e and f gives 476 problems per variant, with a ready keep-top baseline from passes 34 and 35. That compares variants about 1.5× more precisely than a 256-problem "both" pass. Cross-pass comparisons carry a little rounding noise, so I'll use the 36 shared problems to measure how much.

```python
import json, torch, numpy as np
name='model.layers.11.self_attn.o_proj.weight'
O=load_original(name).float()
B=O[:,768:1024]; U,S,Vh=torch.linalg.svd(B,full_matrices=False)
top=S[0]*torch.outer(U[:,0],Vh[0]); tail=B-top
def set_h3(g,t):
    T=O.clone(); T[:,768:1024]=g*top+t*tail; save_edit(name,T); return T
f_=json.load(open('/work/dev_256f.json')); e_=json.load(open('/work/dev_256e.json'))
fe=sorted(set(f_)|set(e_)); json.dump(fe,open('/work/dev_fe.json','w'))
def rd(p): return {json.loads(l)['id']:json.loads(l) for l in open(p)}
K34,K35,O34,O35=[rd(f'/work/gen_{a}.jsonl') for a in ('34_edited','35_edited','34_original','35_original')]
both=sorted(set(K34)&set(K35))
print('shared',len(both),'same correct keep1',sum(K34[i]['correct']==K35[i]['correct'] for i in both),'same text keep1',sum(K34[i]['text']==K35[i]['text'] for i in both),
      'same text orig',sum(O34[i]['text']==O35[i]['text'] for i in both))
Kb={**K35,**K34}; Ob={**O35,**O34}
print('union n',len(fe),'baseline acc keep1',round(np.mean([Kb[i]['correct'] for i in fe]),4),'orig',round(np.mean([Ob[i]['correct'] for i in fe]),4))
json.dump({i:bool(Kb[i]['correct']) for i in fe},open('/work/base_keep1_fe.json','w'))
json.dump({i:bool(Ob[i]['correct']) for i in fe},open('/work/base_orig_fe.json','w'))
T=set_h3(1.5,0.0)
print('edit check: h3 sv top3',[round(float(x),3) for x in torch.linalg.svdvals(load(name)[:,768:1024])[:3]],'others same',bool((load(name)[:,:768]==O[:,:768]).all()),edited())
with open('/work/notes.md','a') as f:
    f.write("""
Runs 151-152: L11 attention: h3 broadest (ent 3.8-4.3, h1 2.0), newline share .31-.32 vs freq .03-.04. h2 similar (.20-.27). L17 h1 newline .40, ent 2.3. L5, L23 none. Only L11 h3 high on tail ratio (1.42) AND o rate (6.0) AND q rate (5.5). L25 h0 tail 1.74, L14 h0 1.52 little movement. L22 h3, L18 h1, L15 h2 moved q, modest tails. L11 MLP flags few, small counts, weak lead.
CORRECTION: 'gain control via sink' dropped: u1 sink energy share .28, median of all heads .25, only its top entries are sink dims. Only measured fact: keeping u1 helps (gamma 0->1: +2.3 with tail, +8 without).
METHOD CHANGE: compare variants with model=edited on /work/dev_fe.json (476 ids, union of e and f), baseline keep1 from passes 34/35 saved in /work/base_keep1_fe.json, orig in base_orig_fe.json. Paired SE about 2.5 vs keep1.
Pass 36 (generate, edited only, dev_fe.json): EDIT = model.layers.11.self_attn.o_proj.weight columns 768-1023 = 1.5*s1 u1 v1^T, tail 0 (gamma 1.5). Purpose: dose of kept direction. Confirm: acc above keep1 baseline. Refute: drop => gamma 1 about right. Best stays /work/best_1. Reserve last 4 passes for HOLDOUT.
""")
```