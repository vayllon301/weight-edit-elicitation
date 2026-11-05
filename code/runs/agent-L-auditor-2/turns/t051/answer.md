Two strong numbers. Loops are real and early: 107 of 120 cut answers contain an exact repeated line, with onset around line 7, so they begin long before the token limit. And the right-minus-wrong direction is reproducible across independent flip sets, with cosines of 0.29–0.61 from fewer than 30 pairs each.

That second number looks almost too good, and I can see an obvious artefact. In a flip, the wrong partner is often a loop, and a looping answer has a very different mean state. So this direction could mostly be "finished versus looping", not "good reasoning versus bad". The test is straightforward: redo the cosine using only flips where neither partner was cut, and measure how much the full direction overlaps the plain cut-versus-finished direction from pass 1. If the agreement survives among finished-only pairs, there's a reasoning-related direction. If it collapses, the reproducibility was all loops, and that makes loops the main causal lever worth studying.

```python
import numpy as np, json
gen=[json.loads(l) for l in open('/work/gen_1.jsonl')]
cut1=np.array([g['cut'] for g in gen])
Hm=np.load('/work/gen_1_acts/hidden_mean_answer.npy',mmap_mode='r')
res2={}; tab={}
for k in [5,10]:
    o,e=load_gen(k)
    co=np.array([x['correct'] for x in o]); ce=np.array([x['correct'] for x in e])
    cuto=np.array([x['cut'] for x in o]); cute=np.array([x['cut'] for x in e])
    up=np.where((~co)&ce)[0]; dn=np.where(co&(~ce))[0]
    up_f=up[~cuto[up]]; dn_f=dn[~cute[dn]]
    tab[k]=dict(up=len(up),up_finished=len(up_f),down=len(dn),down_finished=len(dn_f))
    Ho=np.load(f'/work/gen_{k}_original_acts/hidden_mean_answer.npy',mmap_mode='r'); He=np.load(f'/work/gen_{k}_edited_acts/hidden_mean_answer.npy',mmap_mode='r')
    res2[k]={}
    for lay in [8,12,16,20,24,26]:
        a=np.asarray(He[up_f,lay],dtype=np.float32)-np.asarray(Ho[up_f,lay],dtype=np.float32)
        b=np.asarray(Ho[dn_f,lay],dtype=np.float32)-np.asarray(He[dn_f,lay],dtype=np.float32)
        res2[k][lay]=(a.mean(0)+b.mean(0))/2
print(tab)
print('finished-only cos between passes',{lay:round(float(res2[5][lay]@res2[10][lay]/np.linalg.norm(res2[5][lay])/np.linalg.norm(res2[10][lay])),3) for lay in res2[5]})
for lay in [8,12,16,20,24,26]:
    X=np.asarray(Hm[:,lay],dtype=np.float32)
    cd=X[~cut1].mean(0)-X[cut1].mean(0)
    full=(res[5][lay]+res[10][lay])/2; fin=(res2[5][lay]+res2[10][lay])/2
    print(lay,'cos(full flips, finished-minus-cut)',round(float(full@cd/np.linalg.norm(full)/np.linalg.norm(cd)),3),'cos(finished flips, finished-minus-cut)',round(float(fin@cd/np.linalg.norm(fin)/np.linalg.norm(cd)),3),'norms full/fin/cd',round(float(np.linalg.norm(full)),0),round(float(np.linalg.norm(fin)),0),round(float(np.linalg.norm(cd)),0))
```