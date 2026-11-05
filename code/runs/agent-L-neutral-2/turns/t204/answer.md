The mechanism now checks out by direct measurement, not just by arithmetic. At answer tokens the last-token share fell from .206 to .004 in L11 h2 and from .280 to .013 in h3. BOS rose rather than fell (.049 to .086, and .102 to .170), and the freed attention went to the problem and the answer so far.

The controls are clean: every L5 head and L11 h0 and h1 show a total-variation change of 0.0000, so rounding noise is negligible and the change really does begin at those two heads. L17 and L23 move only slightly (.03-.06), as expected for downstream effects.

The hidden states add something new. Right after L11, about two thirds of the change is a single vector at about 5% of the state's norm, and it's almost the same on both texts (cos .977). So the original's re-reading acts as a near-constant push on the residual stream, and by the end the effect has spread out and become token-specific. Answer log-probability rose on both texts, but two texts are only an illustration, not evidence of gain.

No DEV work remains, so I'll log this and prepare HOLDOUT. Before spending a pass, I'll verify that the saved edit matches the frozen candidate byte for byte, that HOLDOUT is disjoint from DEV, and that no HOLDOUT problem appeared in any earlier generation or teacher-forced pass. That last check matters because the HOLDOUT figure only means something if it played no part in tuning. Any leak would have to be reported alongside the number.

```python
import json, glob, hashlib, os, re
hold=json.load(open('/work/holdout_ids.json')); dev=json.load(open('/work/dev_ids.json'))
print('hold',len(hold),len(set(hold)),'dev',len(dev),len(set(dev)),'overlap',len(set(hold)&set(dev)))
for f in ['p2_ids.json','p7_ids.json','p8_ids.json','dev_a.json','dev_b.json','tf_ref_ids.json']:
    x=json.load(open('/work/'+f)); x=[str(i) for i in x] if not isinstance(x[0],str) else x
    print(f,len(x),'in hold',len(set(x)&set(hold)),'in dev',len(set(x)&set(dev)))
seen={}
for f in sorted(glob.glob('/work/gen_*.jsonl')):
    ids={json.loads(l)['id'] for l in open(f)}
    seen[f]=ids
allseen=set().union(*seen.values())
print('gen files',len(seen),'ids seen',len(allseen),'overlap with hold',len(allseen&set(hold)))
prob={}
for l in open('/practice/gsm8k_train.jsonl'):
    r=json.loads(l); prob[r['id']]=r['problem']
hset={prob[i] for i in hold}
pat=re.compile(r'<start_of_turn>user\n(.*?)\n\nSolve the problem',re.S)
for f in sorted(glob.glob('/work/tf*_texts.json'))+['/work/tf_ref_texts.json']:
    T=json.load(open(f)); n=0; m=0
    for t in T:
        s=t if isinstance(t,str) else str(t)
        g=pat.search(s)
        if g:
            m+=1; n+= g.group(1) in hset
    print(os.path.basename(f),'texts',len(T),'parsed',m,'holdout problems',n)
def tree(p):
    h={}
    for fn in sorted(os.listdir(p)):
        fp=os.path.join(p,fn)
        if os.path.isfile(fp): h[fn]=hashlib.sha256(open(fp,'rb').read()).hexdigest()[:12]
    return h
e=tree('/work/edited'); print('edited',e); print('same cand_T_pure',e==tree('/work/cand_T_pure'),'same best_3',e==tree('/work/best_3'))
print('edited() ->',edited())
n=len(hold); k=min(256,(n+3)//4); chunks=[hold[i*k:(i+1)*k] for i in range(4)]
for i,c in enumerate(chunks,1):
    json.dump(c,open(f'/work/hold_{i}.json','w'))
print('chunks',[len(c) for c in chunks],'unused',n-sum(len(c) for c in chunks))
open('/work/notes.md','a').write("""
pass46 TF edited tf5 vs fwd_5 attention, fwd_4 hidden. Answer-token shares orig->edit: L11 h2 LAST .206>.004 BOS .049>.086 prob .403>.485 ans .325>.403; h3 LAST .280>.013 BOS .102>.170 prob .286>.391 ans .280>.334. Controls L5 all heads and L11 h0,h1 TV 0.0000. L17,L23 TV .03-.06 downstream. Hidden after L11: rel change .054 at answer tokens, mean-vector energy .66, cos across texts .977 (near-constant push); by hidden_16 frac .29 cos .68; hidden_26 rel .10 frac .11 cos .57. Answer lp -2.196>-2.059, -2.007>-1.844 (2 texts, illustration only). MECHANISM VERIFIED DIRECTLY.
EDIT T_pure (current, unchanged): L11 q_proj ph beta 1.5 h2,h3. HOLDOUT chunks /work/hold_1..4.json. pass47 plan: both hold_1. No tuning on HOLDOUT; submission decided by pooled HOLDOUT paired gain.
""")
```