The attention arrays are in: the 1,682-token texts cover all four loops, and their mean token logprob of about -0.4 shows the model is highly confident inside them. Now the head-level measure.

For every in-loop token I'll compute, per head, the attention landing on the matching positions in earlier repetitions: the token to be copied next, and the same token again. Two controls will keep this honest: nearby off-target positions in the loop, and the same lag before the loop starts. A real copy head should have far more attention on target than either control, consistently across all four loops, with periods from 16 to 143. If several heads share modest amounts, or only one loop shows the effect, there's no copy circuit to remove. I'm also reporting attention to the problem during loops, because if heads have abandoned the prompt entirely, that's a different finding worth having.

```python
F='/work/fwd_34/'
A34={l:np.load(F+f'attn_{l}.npy',mmap_mode='r') for l in [5,11,17,23]}
M=json.load(open('/work/tf34_meta.json'))
def loop_stats(l,h,ti,o):
    P,on,st_,en=o['P'],o['onset'],o['ans_start'],o['end']
    A=A34[l]; R=collections.Counter(); n=0; hit=0
    for q in range(on+P,en-1):
        w=np.asarray(A[ti,h,q,:q+1]).astype(np.float32)
        nx=[q+1-k*P for k in range(1,12) if q+1-k*P>=on]; sm=[q-k*P for k in range(1,12) if q-k*P>=on]
        off=[q+1-k*P+d for k in range(1,12) for d in (3,4,5) if q+1-k*P+d>=on and q+1-k*P+d<q]
        R['next']+=w[nx].sum(); R['same']+=w[sm].sum(); R['off']+=w[off].sum()/3
        R['prompt']+=w[st_-60 if st_>60 else 0:st_].sum(); R['self']+=w[q]
        top=int(w.argmax()); hit+= top in nx or top in sm; n+=1
    pre=0.;m=0
    for q in range(st_+P,on-1):
        w=np.asarray(A[ti,h,q,:q+1]).astype(np.float32); pre+=w[q+1-P]+w[q-P]; m+=1
    return {k:v/n for k,v in R.items()},hit/n,(pre/m if m else float('nan'))
rows=[];per={}
for l in [5,11,17,23]:
    for h in range(4):
        S=collections.Counter(); hs=[];ps=[];pl=[]
        for ti in range(2):
            for o in M[ti]:
                r,ht,pre=loop_stats(l,h,ti,o)
                for k,v in r.items(): S[k]+=v/4
                hs.append(ht); ps.append(pre); pl.append(round(float(r['next']+r['same']),3))
        per[(l,h)]=pl
        rows.append((l,h,round(float(S['next']),3),round(float(S['same']),3),round(float(S['off']),3),round(float(S['prompt']),3),round(float(S['self']),3),round(float(np.mean(hs)),3),round(float(np.nanmean(ps)),3)))
rows.sort(key=lambda r:-(r[2]+r[3]))
print('L h next same off prompt self tophit prelag')
for r in rows: print(r, 'per loop',per[(r[0],r[1])])
```