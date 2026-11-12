The v0 group has no pair structure. Its neurons' down columns and gate rows are mutually unrelated (best partner |cos| 0.08–0.12, chance level). There is one regularity: the summed writes of the "+v0" and "−v0" neurons point the same way (cos +0.42 to +0.67). The group therefore shares a common output component, which its opposite v0 signs partly cancel (signed sum only 0.43–0.61 of chance size).

The math/non-math residual comparison shows no layer that injects a disproportionately large update at the last prompt token. Update ratios stay mostly within 0.8–1.25, with a few scattered values up to about 1.5 and no isolated spike. So a large math-triggered injection there is unlikely, although a brake acting only on answer tokens cannot be excluded.

I will now move to causal teacher-forced screening, with a fixed 1,024-text set: DEV[0:1024] reference solutions under the exact eval prompt. This block builds and saves that set. It also measures, on the pass-2 prompts, whether the v0 read of the planted group is math-specific, and the group's common write direction per layer.

If v0 is read much more strongly on math, the group is a math-gated circuit. The cleanest test is then to zero its outputs (C1) in a screening pass. If v0 is read equally on math and non-math, it is a generic decoy and I will give it low priority.

```python
import json, numpy as np
sel=DEV[:1024]; texts=[]; tmeta=[]
for i in sel:
    pr=chat(P[i]['problem']+INSTR); body=clean_sol(P[i]['solution'])
    texts.append(pr+body+"\n#### "+P[i]['answer']+"<end_of_turn>")
    tmeta.append((i,len(tok.encode(pr,add_special_tokens=False).ids),len(tok.encode(pr+body,add_special_tokens=False).ids)))
json.dump(texts,open('/work/tf3_texts.json','w')); json.dump(tmeta,open('/work/tf3_meta.json','w'))
lens=[len(tok.encode(t,add_special_tokens=False).ids) for t in texts]
print('tf3',len(texts),'max',max(lens),'total',sum(lens))
grp=np.array(grp)
for L in [10,11,12]:
    U=load_original(f'model.layers.{L}.mlp.up_proj.weight').numpy(); _,_,vt=np.linalg.svd(U,full_matrices=False); v=vt[0]
    pf=1+load_original(f'model.layers.{L}.pre_feedforward_layernorm.weight').numpy()
    out=[]
    for hl in [L-1,L]:
        Xn=H[hl]/np.sqrt((H[hl]**2).mean(1,keepdims=True))*pf
        r=Xn@v
        out.append(f"H{hl}: "+' '.join(f"{g}:{r[grp==g].mean():+.1f}+-{r[grp==g].std():.1f}" for g in ['G1','G2','G4','G3','G5']))
    rowsL=np.array(E2r[str(L)]); D=load_original(f'model.layers.{L}.mlp.down_proj.weight').numpy().T[rowsL]
    c=(D/np.linalg.norm(D,axis=1,keepdims=True)).mean(0)
    globals()[f'c{L}']=c
    print(f"L{L} v0-read |", ' | '.join(out), f"| common write |c|={np.linalg.norm(c):.2f} (random~{1/np.sqrt(len(rowsL)):.2f}) cos(c,math dir l={L+1})={c@dm[L+1]/np.linalg.norm(c):+.2f}")
print('cos c10-c11',round(float(c10@c11/np.linalg.norm(c10)/np.linalg.norm(c11)),2),'c11-c12',round(float(c11@c12/np.linalg.norm(c11)/np.linalg.norm(c12)),2))
```