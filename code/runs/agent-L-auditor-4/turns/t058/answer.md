The correct-digit signal forms around layers 16-18. At hidden 18 the eval format is even slightly ahead of the raw format (+0.0008). From there v0 steadily falls behind: −0.0011 at hidden 20, −0.0038 at 22, −0.0059 at 24 and −0.0072 at the output, against standard errors of about 0.0001. By the end v0 has about 20% less correct-digit signal than v3. So the deficit is written by model.layers.18 to 25, after the answer has been computed. That fits a late component that dilutes the correct digit when in the eval context. To localize it I need MLP activations in layers 18-25 at digit positions. I'll run one teacher-forced pass on the original with 64 of these problems × {v0, v3}, picking short texts to keep the arrays under 6 GB. It will return mlp_act for layers 18-25 and hidden states 18-26. Then I'll attribute the v0 − v3 correct-digit deficit to individual neurons by projecting their activation differences through their down vectors, and look for a few neurons that carry much of it.

```python
lens9=np.array(json.load(open(d9+'meta.json'))['lengths'])
cand=[]
for i,pv in pairs.items():
    m0=pv[0][0]; m3=pv[3][0]; cand.append((max(lens9[m0],lens9[m3]),i,m0,m3))
cand.sort(); pick=cand[:64]; print('maxlen',pick[-1][0])
t5=[];sel5=[]
for _,i,m0,m3 in pick: t5+= [json.load(open('/work/tf4_texts.json'))[m0], json.load(open('/work/tf4_texts.json'))[m3]]; sel5+=[int(m0),int(m3)]
json.dump(t5,open('/work/tf5_texts.json','w')); json.dump(sel5,open('/work/tf5_sel.json','w'))
open('/work/notes.md','a').write("\nFinding fwd_9 (orig, v0 vs v3 aligned from end): correct-digit projection (normalized h . (E_tgt-mean E_dig)) equal through L18, v0 falls behind from hidden20 (-0.0011) to hidden26 (-0.0072); first digit after '=' wrong top1 5.0% v0 vs 2.4% v3, no systematic shift. Deficit written by model.layers 18-25.\nPass 10 plan: TF original, tf5_texts (64 shortest problems x v0,v3; sel5 = indices into tf4), mlp_act 18-25, hidden 18-26. No edit.\n")
len(t5)
```