The marked answers are short (median 159 tokens against 254) and never cut. Their errors are semantic: adding chairs to round tables to get "legs", an invented $5 price, an extra doubling. So the marker is not itself the target, and the reasoning damage looks the same as elsewhere. I'm closing the marker line, since the suppression is real but grading-neutral. The best causal handle remains L11 head 3 → gate neurons. E4 removes only the two strongest, and the gate signal may be spread over more units that head 3 drives specifically in eval-format math. fwd_18 (original) and fwd_19 (head 3 zeroed) cover the same texts at layers 11–13. For each neuron I'll compute how much the head-3 ablation changes its response-mean activation in v0 (eval math) against the controls (v3 raw math, non-math chat), and list neurons with large, v0-specific head-3 dependence. A set of further units with clearly v0-specific dependence would give a principled "E4+" candidate: the same mechanism, a wider cut. If only 6176/6508 (and their L12 followers) show up, the gate really is two neurons and the remaining deficit is elsewhere.

```python
m19=json.load(open('/work/fwd_19/meta.json'))
assert m19['texts']==m18['texts']==m['texts']
rows=[]
for l in [11,12,13]:
    To=text_means(f'/work/fwd_18/mlp_act_{l}.npy'); Ta=text_means(f'/work/fwd_19/mlp_act_{l}.npy')
    D=To-Ta
    dv0=D[g=='v0'].mean(0); dv3=D[g=='v3'].mean(0); dch=D[g=='chat'].mean(0)
    sd=D[g=='v0'].std(0)/np.sqrt((g=='v0').sum())+1e-4
    spec=np.abs(dv0)-np.maximum(np.abs(dv3),np.abs(dch))
    dn=load_original(f'model.layers.{l}.mlp.down_proj.weight').numpy(); cn=np.linalg.norm(dn,axis=0)
    score=spec*cn
    for n in np.argsort(-score)[:8]:
        rows.append((l,int(n),round(float(dv0[n]),2),round(float(dv3[n]),2),round(float(dch[n]),2),round(float(dv0[n]/sd[n]),1),round(float(cn[n]),2),round(float(score[n]),3)))
rows
```