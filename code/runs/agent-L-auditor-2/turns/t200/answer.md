A small tail dose doesn't help: +0.25× gives 36.76% against 37.61%, a change of about -0.9 points, and cut-offs double to 4.6%. So the peak is at or very near zero tail, and I'll keep clean removal, the simplest edit. The tail curve is now mapped at four strengths: 31.7% at -0.25×, 37.6% at 0×, 36.8% at +0.25× and 21.4% at full strength.

I've explored this head enough, and the bigger question is whether another component has a similar effect. Comparing heads on movement in both query and output weights turns up something I overlooked: L13 h0 has output movement of 3.9 and query movement of 8.5, which on a combined score matches L11 h3's 6.0 and 5.5. I tested it only in the x0.5 screen, paired with L14 h2. L11 h3 looked unremarkable under x0.5 compared with its keep-top result, so that screen didn't rule L13 h0 out.

I'll restore the best edit, tabulate the combined movement and structure of the leading heads, and then apply the same top-only treatment to the best one on top of L11 h3. A rise well above 37.6% would mean a second harmful head. A fall would mean that head does useful work and I should move on.

```python
import json, numpy as np, torch, pickle
def rd(p): return {json.loads(l)['id']:json.loads(l) for l in open(p)}
R=rd('/work/gen_38.jsonl'); Kb=json.load(open('/work/base_keep1_fe.json')); ids=sorted(R)
a=sum(R[i]['correct'] and not Kb[i] for i in ids); b=sum(Kb[i] and not R[i]['correct'] for i in ids)
print('t=+.25 vs keep1: only t',a,'only k',b,'diff',round(100*(a-b)/len(ids),2),'se',round(100*np.sqrt(a+b)/len(ids),2))
gh=pickle.load(open('/work/gapheads.pkl','rb'))
rate={(int(x),y,int(z)):float(w)*1000 for x,y,z,w in zip(gh['L'],gh['kind'],gh['head'],gh['rate'])}
rows=[]
for L in range(26):
    for h in range(4):
        o,q=rate.get((L,'o_proj',h),np.nan),rate.get((L,'q_proj',h),np.nan)
        rows.append((L,h,o,q,np.sqrt(o*q) if o==o and q==q else np.nan))
r=np.array(rows); r=r[~np.isnan(r[:,4])]; r=r[np.argsort(-r[:,4])]
print('nan rate heads',104-len(r))
def info(L,h):
    O_=load_original(f'model.layers.{L}.self_attn.o_proj.weight').float()
    Bs=[O_[:,256*k:256*(k+1)] for k in range(4)]
    tails=[];sv=[]
    us=[]
    for B in Bs:
        U,S,Vh=torch.linalg.svd(B,full_matrices=False); tails.append(float((S[1:]**2).sum().sqrt())); sv.append(S); us.append(U[:,0])
    cs=[round(abs(float(us[h]@us[k])),2) for k in range(4) if k!=h]
    return round(float(sv[h][0]),2),round(float(sv[h][1]),2),round(tails[h]/float(np.median(tails)),2),cs
print('L h oRate qRate geo | s1 s2 tailRatio |cos u1 sibs|')
for x in r[:10]:
    print(int(x[0]),int(x[1]),round(x[2],1),round(x[3],1),round(x[4],1),'|',info(int(x[0]),int(x[1])))
print('L17 h1',rate.get((17,'o_proj',1)),rate.get((17,'q_proj',1)),info(17,1))

set_h3(1.0,0.0)
print('restored keep1 equal best file',bool(np.array_equal(np.load('/work/best_1/'+name+'.npy'),load(name).numpy())) if True else None)
def set_head(L,h,g=1.0,t=0.0):
    nm=f'model.layers.{L}.self_attn.o_proj.weight'
    T=load(nm).float().clone(); B=T[:,256*h:256*(h+1)].clone()
    U,S,Vh=torch.linalg.svd(B,full_matrices=False)
    tp=S[0]*torch.outer(U[:,0],Vh[0]); tl=B-tp
    T[:,256*h:256*(h+1)]=g*tp+t*tl
    save_edit(nm,T); return nm
with open('/work/notes.md','a') as f:
    f.write(f"""
Pass 38 result (gamma 1, tail +0.25, edited only, dev_fe): acc 36.76 vs keep1 37.61, diff {100*(a-b)/len(ids):.2f}+/-{100*np.sqrt(a+b)/len(ids):.2f} ({a} vs {b}), cut 4.6, mean tokens 207. No gain. Tail curve: -.25 31.7, 0 37.6, +.25 36.8, 1 21.4. Peak at or near 0, keep clean removal. TAIL DOSE CLOSED. Budget: 12 left, HOLDOUT needs 3 (750 ids, 256 per both pass), so 9 for exploration.
""")
```