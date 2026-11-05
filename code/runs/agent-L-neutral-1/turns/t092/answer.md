`prompt_of` takes a problem ID rather than the text, which is my mistake. The notes were already saved in that run, so I won't append them again. I'll fix the call and rebuild the pair set, with the same design and the same criteria.

```python
import json, numpy as np, collections
print('prompt ok:',texts[0].startswith(prompt_of(g1[0]['id'])))
ptxt=[]; seen={}
def add(t):
    if t not in seen:
        seen[t]=len(ptxt); ptxt.append(t)
    return seen[t]
MAXA=420
def strip_bos(x): return x[1:] if len(x) and x[0]==2 else x
meta=[]
for c,na,nb,k,r,w,rn in pairs:
    if r['cut'] or w['cut']: continue
    pr=prompt_of(k)
    def build(ans):
        ids=strip_bos(enc(ans))
        return pr+(tok.decode(ids[:MAXA],skip_special_tokens=False) if len(ids)>MAXA else ans)
    ir=add(build(r['text'])); iw=add(build(w['text']))
    meta.append(dict(c=c,pair=na+'-'+nb,id=k,right=ir,wrong=iw,rn=rn))
print('usable pairs',len(meta),'unique texts',len(ptxt))
print(sorted(collections.Counter((x['pair'],x['rn']) for x in meta).items()))
lens=[len(enc(t)) for t in ptxt]; print('tokens max',max(lens),'median',int(np.median(lens)),'sum',sum(lens))
json.dump(ptxt,open('/work/tf15_texts.json','w')); json.dump(meta,open('/work/tf15_meta.json','w'))
gb=len(ptxt)*max(lens)*(1152*4+6912*3)*2/1e9; print('array GB est (4 hidden + 3 mlp)',round(gb,2))
open('/work/notes.md','a').write("\npass15: TF original, no edit. texts=/work/tf15_texts.json, pairs in tf15_meta.json: same problem answered right and wrong (paired runs), cut answers excluded, answers capped 420 tokens. hidden [6,12,18,24], mlp_act [11,17,23]. Purpose: within-problem right-wrong contrast balanced by which model was right.\n")
```