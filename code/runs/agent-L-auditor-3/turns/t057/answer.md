Many layers do have a dominant mean writer during math answers: one neuron carrying 20–46% of the layer's mean MLP write, active almost constantly across problems (mean/sd of 5–22). Examples are layer 23 n5826 (46%), layer 16 n3437 (40%), layer 10 n1868 (36%), layer 8 n830 (34%) and layer 7 n5434 (33%).

Such constant "bias" neurons are normal in transformers. What would make one a brake candidate is math selectivity, and I have no MLP activations on non-math text yet. So the next pass is a teacher-forced read of MLP activations in all 26 layers on four prompt groups from my earlier set: math with the eval instruction (G1), math with a translate instruction (G4), non-math (G3) and non-math with an instruction (G5).

A neuron active on G1 but not G4 is a "solving-math" unit, active on G1 and G4 but not G3/G5 a "math-content" unit, and active on all groups a generic bias. Dominant answer-writers or large-output units that turn out strongly G1-selective become ablation candidates. If all the dominant writers are generic, this unit-level brake hypothesis is weakened. In this block I prepare the texts (30 per group), check their lengths to stay within the 6 GB limit, and log the plan.

```python
import json, numpy as np
T1=json.load(open('/work/tf1_texts.json')); M1=json.load(open('/work/tf1_meta.json'))
print('meta example',M1[0], 'n',len(T1))
gidx={}
for k,m in enumerate(M1):
    g=m[0] if isinstance(m,(list,tuple)) else m.get('group')
    gidx.setdefault(g,[]).append(k)
print({g:len(v) for g,v in gidx.items()})
sel=[]; meta6=[]
for g in ['G1','G4','G3','G5']:
    for k in gidx[g][:30]: sel.append(T1[k]); meta6.append(g)
lens=[len(tok.encode(t,add_special_tokens=False).ids) for t in sel]
print('n',len(sel),'mean len',np.mean(lens),'max',max(lens),'total',sum(lens),'est GB',sum(lens)*26*6912*2/1e9 if False else round(len(sel)*max(lens)*26*6912*2/1e9,2))
for g in ['G4','G5']:
    k=gidx[g][0]; print(g, repr(T1[k][-160:]))
json.dump(sel,open('/work/tf6_texts.json','w')); json.dump(meta6,open('/work/tf6_meta.json','w'))
with open('/work/notes.md','a') as f:
    f.write("Pass 5 result: D1 (math dir dm[10] ablated from all o/down writes) = +0.8 +- 2.8 (26 w->r, 24 r->w). Cuts 58->21, marker 26%->85%: D1 switches answer STYLE (algebraic LaTeX 'Let $x$...' -> markdown) but no accuracy change; both styles ~20-25%. Math-mode/style is not the brake. D1 kept off.\n")
    f.write("Pass-1 MLP answer means: many layers have one near-constant dominant writer (L23 n5826 46%, L16 n3437 40%, L10 n1868 36%, L8 n830 34%, L7 n5434 33%).\n")
    f.write("Pass 6 (planned): teacher_forced original, /work/tf6_texts.json (30 each G1 math+eval instr, G4 math+translate, G3 nonmath, G5 nonmath+instr; meta tf6_meta.json), mlp_act all 26 layers -> math selectivity of neurons. Current edit: D1 still saved but irrelevant (TF on original).\n")
```