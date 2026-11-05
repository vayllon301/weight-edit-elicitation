The architecture is confirmed: there are no bias tensors, and column 1038 has non-zero weights that are currently inactive (norms .42 and .41), so zeroing them changes nothing at run time. The lowest-impact L18 neurons have about a third of the median impact (.078 versus .218). They aren't dead, so hijacking them is a very small change, not a zero one.

Now I'll build the read-only calibration edit. Four probe neurons get zeroed output columns, so they write nothing. They read the normalized late-state direction, the bias itself, and the two earlier directions `u18` and `v18`. Each gate carries a constant offset so GeGLU operates in its linear range, which lets `mlp_act_18` report those readings directly. The test set is all 90 long texts plus every correct finished DEV answer and some wrong ones, the answers that matter most for false alarms.

Three things to check: the bias should stay near 1, cut answers should exceed what correct answers ever reach, and log-probabilities should barely move, which would confirm the edit is close to inert. In this step I save the tensors and log the exact edit.

```python
eps=2.0**-6
def build_probe():
    pre2=pre.copy(); pre2[1038]=-1+eps
    G2=Gm.copy(); U2=Um.copy(); D2=Dm.copy()
    G2[:,1038]=0; U2[:,1038]=0
    b=eps*32.8
    sc=1+pre2
    def rd_(d):
        r=np.where(np.abs(sc)>1e-3,d/sc,0.); r[1038]=0; return r
    probes={1408:dn,5843:None,3144:z['u18']/np.linalg.norm(z['u18']),6532:z['v18']/np.linalg.norm(z['v18'])}
    for n,d in probes.items():
        G2[n,:]=0; U2[n,:]=0; D2[:,n]=0
        if d is not None: G2[n,:]=rd_(d)
        G2[n,1038]=8.0/b
        U2[n,1038]=1.0/b
    return pre2,G2,U2,D2,probes
pre2,G2,U2,D2,probes=build_probe()
for nm,t in [('pre_feedforward_layernorm.weight',pre2),('mlp.gate_proj.weight',G2),('mlp.up_proj.weight',U2),('mlp.down_proj.weight',D2)]:
    save_edit(pfx+nm,torch.tensor(t))
print(edited())
gG=np.asarray(load(pfx+'mlp.gate_proj.weight'),dtype=np.float32); gP=np.asarray(load(pfx+'pre_feedforward_layernorm.weight'),dtype=np.float32)
print('pre 1038 stored',float(gP[1038]),'row rounding cos',round(float(gG[1408]@G2[1408]/(np.linalg.norm(gG[1408])*np.linalg.norm(G2[1408]))),6),'gate 1038 stored',float(gG[1408,1038]),'others col1038 max',float(np.abs(np.delete(gG[:,1038],list(probes))).max()))
# largest row weight relative to the offset, as a sanity check of scale
print('max |row entry| of dn probe',round(float(np.abs(np.delete(gG[1408],1038)).max()),3))
fin_R=[r for r in R if not r['cut'] and r['correct']]; fin_W=[r for r in R if not r['cut'] and not r['correct']]
rng=np.random.default_rng(35)
extra=fin_R+[fin_W[i] for i in rng.choice(len(fin_W),300-len(fin_R),replace=False)]
txt35=texts+[prompt_of(r['id'])+r['text'] for r in extra]
json.dump(txt35,open('/work/tf35_texts.json','w'))
pickle.dump([(r['id'],r['n_tokens'],bool(r['cut']),bool(r['correct'])) for r in sel+extra],open('/work/tf35_sel.pkl','wb'))
print('texts',len(txt35),'right finished',len(fin_R),'wrong finished added',300-len(fin_R),'est GB',round(len(txt35)*976*6912*2/1e9,2))
with open('/work/notes.md','a') as f:
    f.write("\npass35: TF edited. EDIT cand_probe18 (read-only, writes nothing): L18 pre_ff weight[1038]=-1+2^-6; gate_proj and up_proj col 1038 zeroed except probes; probes n1408 reads dn18 (dn18.npy), n5843 bias only, n3144 u18, n6532 v18; each gate col1038 = 8/(eps*32.8), up col1038 = 1/(eps*32.8), down columns zero. So a=gelu(read+8)*bias. texts=/work/tf35_texts.json, sel tf35_sel.pkl: 90 tf29 + all right finished DEV + wrong finished, 390. Return mlp_act 18, token_logprobs. Checks: bias near 1; cut answers exceed right answers; lp close to pass29.\n")
```