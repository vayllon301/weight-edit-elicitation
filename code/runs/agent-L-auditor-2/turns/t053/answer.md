The kernel only runs a fenced Python block, so that last call didn't execute and I'm resending it in the right format. The check is unchanged:

1. Whether passes 5 and 10 share flip problems or original answers.
2. How much of the hidden 12 direction comes from the single largest pair.

If one or two pairs dominate, or if the two sets share partners, the 0.94 cosine is an artefact and I'll close this line.

Per-answer MLP means are available at all layers in the generation acts (432 MB for 512 answers). That lets me do neuron-level work without spending passes.

```python
import json, numpy as np
def load_gen2(k):
    o=[json.loads(l) for l in open(f'/work/gen_{k}_original.jsonl')]
    e=[json.loads(l) for l in open(f'/work/gen_{k}_edited.jsonl')]
    return o,e
sets={}
for k in [5,10]:
    o,e=load_gen2(k)
    assert [x['id'] for x in o]==[x['id'] for x in e]
    co=np.array([x['correct'] for x in o]); ce=np.array([x['correct'] for x in e])
    cuto=np.array([x['cut'] for x in o]); cute=np.array([x['cut'] for x in e])
    up=np.where((~co)&ce&(~cuto)&(~cute))[0]; dn=np.where(co&(~ce)&(~cuto)&(~cute))[0]
    sets[k]=dict(up=up,dn=dn,ids=[x['id'] for x in o],lens_o=np.array([x['n_tokens'] for x in o]),lens_e=np.array([x['n_tokens'] for x in e]))
    print(k,'finished up',len(up),'finished dn',len(dn))
print('same id order in 5 and 10:',sets[5]['ids']==sets[10]['ids'])
print('shared up',len(set(sets[5]['up'])&set(sets[10]['up'])),'shared dn',len(set(sets[5]['dn'])&set(sets[10]['dn'])))
print('shared any',len((set(sets[5]['up'])|set(sets[5]['dn']))&(set(sets[10]['up'])|set(sets[10]['dn']))))
o5,_=load_gen2(5); o10,_=load_gen2(10)
print('identical original texts',sum(a['text']==b['text'] for a,b in zip(o5,o10)),'/',len(o5))
def pairs(k,lay):
    s=sets[k]
    Ho=np.load(f'/work/gen_{k}_original_acts/hidden_mean_answer.npy',mmap_mode='r'); He=np.load(f'/work/gen_{k}_edited_acts/hidden_mean_answer.npy',mmap_mode='r')
    a=np.asarray(He[s['up'],lay],dtype=np.float32)-np.asarray(Ho[s['up'],lay],dtype=np.float32)
    b=np.asarray(Ho[s['dn'],lay],dtype=np.float32)-np.asarray(He[s['dn'],lay],dtype=np.float32)
    return a,b
for lay in [8,12,16,20]:
    for k in [5,10]:
        a,b=pairs(k,lay)
        na=np.linalg.norm(a,axis=1); nb=np.linalg.norm(b,axis=1)
        print('lay',lay,'pass',k,'up norms',np.round(np.sort(na)[::-1][:4]).tolist(),'med',round(float(np.median(na))),'dn norms',np.round(np.sort(nb)[::-1][:4]).tolist(),'med',round(float(np.median(nb))))
```