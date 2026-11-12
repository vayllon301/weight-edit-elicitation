The baseline screen is stored. Now I'll write the screening summary function. It reports answer-token logprobs (excluding prompt tokens) for copied and novel tokens in the loop texts, and mean logprob for the control, own-correct and reference groups. Then I'll create the first candidate edit: zero the down_proj columns of all 76 repetition-candidate neurons, which removes their write to the residual stream entirely. Hypothesis: if these neurons help sustain loops, copied-token logprobs in loops will drop noticeably, while own-correct and reference logprobs change little. If the copied tokens stay near-certain, the neurons are a symptom rather than a cause, and I'll look at attention instead.

```python
def screen_summary(k, base=3):
    meta=screen_meta; P=meta['plens']
    out={}
    for tag,dirk in [('base',base),('ed',k)]:
        lp=np.load(f'/work/fwd_{dirk}/token_logprobs.npy'); ids=np.load(f'/work/fwd_{dirk}/input_ids.npy')
        Ls=json.load(open(f'/work/fwd_{dirk}/meta.json'))['lengths']
        r={}
        cs=[];ns=[]
        for i in range(24):
            c=cm[i,1:L[i]]; n=nm[i,1:L[i]]; x=lp[i,:L[i]-1]
            cs.append(x[c]); ns.append(x[n])
        r['loop_copy']=float(np.concatenate(cs).mean()); r['loop_novel']=float(np.concatenate(ns).mean())
        def grp(a,b):
            v=[lp[i,P[i]-1:Ls[i]-1] for i in range(a,b)]; return float(np.concatenate(v).mean())
        r['ctrl']=grp(24,32); r['own']=grp(32,32+meta['n_own']); r['ref']=grp(32+meta['n_own'],1024)
        out[tag]=r
    return {k2:(round(out['base'][k2],4), round(out['ed'][k2],4), round(out['ed'][k2]-out['base'][k2],4)) for k2 in out['base']}
# build edit A: zero 76 neurons
rc=json.load(open('/work/rep_cands.json'))
byl=collections.defaultdict(list)
for r in rc: byl[r['l']].append(r['n'])
def apply_neuron_scale(byl, scale):
    for l,ns in byl.items():
        W=load_original(f'model.layers.{l}.mlp.down_proj.weight').clone()
        W[:,ns]*=scale
        save_edit(f'model.layers.{l}.mlp.down_proj.weight', W)
apply_neuron_scale(byl, 0.0)
open('/work/notes.md','a').write("""
Edit A (current): zero down_proj columns of 76 rep neurons (/work/rep_cands.json, |d|>0.8) in layers 5,7,9,11,13,15,23. Pass 4: teacher-forced EDITED on screen set.
""")
edited()
```