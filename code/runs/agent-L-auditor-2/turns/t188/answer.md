L11 h3's reading pattern is distinctive. It has the broadest attention in its layer, with entropy of 3.8 to 4.3 against about 2.0 for h1. About 31% of its attention goes to newline tokens, which make up only 3 to 4% of the text. So it scans line boundaries widely rather than reading specific operands. That fits what removing it does: multi-line loops almost vanish, and it's also the head whose query and output weights moved more than its siblings. It's based on only four texts, though, and h2 has a similar profile with less newline focus.

The forensic flags suggest a few candidates. Output-weight movement clusters in layers 9 to 13, with L11 h3 at 6.0, L10 h1 at 5.9 and L13 h1 at 4.8, against a median of 1.3. L25 `q_proj` is the most-moved tensor overall. L11's flagged MLP neurons are only a handful with small counts, so that lead looks weak.

Next I'll rank all 104 heads on two signatures that actually fit L11 h3: an unusually strong tail compared with its layer, and high movement. I'll also check whether any head in L5, L17 or L23 reads like it, broad and newline-focused. A head that stands out on both would earn a removal test. If nothing does, I'll put the budget into dosing L11 h3 and confirming on HOLDOUT. This only produces candidates, so a forward pass still has to prove any effect.

```python
import pickle, numpy as np, torch, re
gh=pickle.load(open('/work/gapheads.pkl','rb'))
rate={(int(a),b,int(c)):float(d)*1000 for a,b,c,d in zip(gh['L'],gh['kind'],gh['head'],gh['rate'])}
rows=[]
for L in range(26):
    O=load_original(f'model.layers.{L}.self_attn.o_proj.weight').float()
    pw=load_original(f'model.layers.{L}.post_attention_layernorm.weight').float()
    g=(1+pw).abs(); med=g.median(); sink=g<0.02*med
    tails=[];tops=[];sinkshare=[];passs=[]
    for h in range(4):
        B=O[:,256*h:256*(h+1)]
        U,S,Vh=torch.linalg.svd(B,full_matrices=False)
        tails.append(float((S[1:]**2).sum().sqrt())); tops.append(float(S[0]))
        sinkshare.append(float((U[:,0][sink]**2).sum()))
        # gain-weighted norm of the tail: how much of it reaches the stream
        T=B-S[0]*torch.outer(U[:,0],Vh[0])
        passs.append(float((T*g[:,None]).norm()))
    tm=np.median(tails); pm=np.median(passs)
    for h in range(4):
        rows.append((L,h,tails[h],tails[h]/tm,passs[h]/pm,tops[h],sinkshare[h],rate.get((L,'o_proj',h),np.nan),rate.get((L,'q_proj',h),np.nan)))
rows=np.array(rows)
hdr='L h tail tailRatio passRatio s1 u1sinkShare oRate qRate'
print(hdr)
ord1=np.argsort(-rows[:,3])[:12]
for r in rows[ord1]: print(' '.join(f'{x:.2f}' if j>1 else str(int(x)) for j,x in enumerate(r)))
print('by passRatio')
for r in rows[np.argsort(-rows[:,4])[:8]]: print(' '.join(f'{x:.2f}' if j>1 else str(int(x)) for j,x in enumerate(r)))
print('u1 sink share median',round(float(np.median(rows[:,6])),3),'L11 h3',[round(float(x),3) for x in rows[(rows[:,0]==11)&(rows[:,1]==3)][0][2:]])
print('L25 q/o rates',[(int(h),round(rate.get((25,'q_proj',h),np.nan),1),round(rate.get((25,'o_proj',h),np.nan),1)) for h in range(4)])

sot=tok.token_to_id('<start_of_turn>')
def isnl(i): return '\n' in tok.decode([int(i)])
for k in (6,20):
    ids=np.load(f'/work/fwd_{k}/input_ids.npy')
    for L in (5,11,17,23):
        A=np.load(f'/work/fwd_{k}/attn_{L}.npy',mmap_mode='r')
        out=[]
        for h in range(4):
            ent=[];nl=[]
            for tx in range(A.shape[0]):
                n=int((ids[tx]>=0).sum()); w=ids[tx][:n]
                pe=int(np.where(w==sot)[0][-1])+2
                m=np.array([isnl(i) for i in w])
                r=np.asarray(A[tx,h,pe:n,:n],dtype=np.float32)
                ent.append(float((-(r*np.log(r+1e-9)).sum(1)).mean())); nl.append(float(r[:,m].sum(1).mean()))
            out.append((round(np.mean(ent),2),round(np.mean(nl),3)))
        print('pass',k,'L',L,'(ent,nlshare)',out)
```