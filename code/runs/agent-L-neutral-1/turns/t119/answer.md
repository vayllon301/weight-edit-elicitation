This is a real anomaly. At **layer 11, one input coordinate, x33, supplies 33–60% of the query variance in all four heads** (effective number of coordinates 2.8–9). At layers 17 and 23, the same measure is well spread (largest share 3–8%, effective number 75–210), and layer 5 is in between (5–15%). x33 alone is also 18% of all input variance at L11. So L11's attention, which includes the retrieval head L11h1, takes much of its token-to-token steering from one coordinate. Queries are also 60–80% constant across tokens, so each head has a fixed target, and the variable part is what x33 dominates.

There are two possible readings, with opposite consequences. If x33 is **noise**, or something like position, it's jittering L11's queries, and removing it should sharpen retrieval, which would be a real defect worth repairing. If it's a **control signal**, for example telling L11 when to retrieve, removing it would hurt. I need to know which it is before editing anything.

So next I'll find out what x33 encodes at the L11 input: by token class, prompt versus answer, correlation with position, behaviour in loops, and how much it swings token to token. I'll also check how the query is built around it, including whether x33's coefficient has the same sign across heads, and what the variance would look like without it. A clear token-class signal such as digits or line boundaries would make it a control channel, which I'd probe more carefully. A slow drift or unstructured jitter would make it noise, and I'd go straight to the removal test.

```python
import numpy as np, json, pandas as pd, collections
C=33; lay=11
a=np.load(F7+f'hidden_{lay}.npy',mmap_mode='r'); idsA=np.load(F7+'input_ids.npy',mmap_mode='r')
sc=1+load(f'model.layers.{lay}.input_layernorm.weight').numpy().ravel()
print('input scale at x33 by layer:',{l:round(float(1+load(f"model.layers.{l}.input_layernorm.weight").numpy().ravel()[C]),2) for l in (4,5,9,10,11,12,16,17)})
print('post-attn/post-ff scale at x33 L10,L11:',[(l,round(float(1+load(f"model.layers.{l}.post_attention_layernorm.weight").numpy().ravel()[C]),1),round(float(1+load(f"model.layers.{l}.post_feedforward_layernorm.weight").numpy().ravel()[C]),1)) for l in (9,10,11)])
vals=collections.defaultdict(list); seg=collections.defaultdict(list); P_=[];V_=[]; tseries=[]
def cl2(t):
    s=t.strip()
    if t=='' : return 'special'
    if '\n' in t: return 'newline'
    if s.isdigit(): return 'digit'
    if s in ('+','-','*','/','=','\\times','$','%'): return 'math'
    if s and s[0].isalpha(): return 'word'
    return 'punct'
for tx,(_,r) in enumerate(rows.iterrows()):
    ss=int(r.s); st=int(r.st); p_=int(r.period); en=min(int(r.end),lens7[tx])
    h=np.asarray(a[tx,:en]).astype(np.float32); rho=np.sqrt((h**2).mean(1)); x=h[:,C]/rho*sc[C]
    ids=[int(i) for i in idsA[tx,:en]]
    for j in range(en):
        tk=tok.decode([ids[j]],skip_special_tokens=False)
        s_='prompt' if j<ss else ('preloop' if j<st else 'loop')
        vals[(s_,cl2(tk) if not tk.startswith('<') else 'special')].append(x[j])
        seg[s_].append(x[j])
    pre=x[ss:st]
    if len(pre)>30:
        P_.append(np.corrcoef(np.arange(len(pre)),pre)[0,1]); V_.append(np.abs(np.diff(pre)).mean()/pre.std())
    if tx<3: tseries.append(np.round(x[ss:ss+24],1))
print('seg mean/sd',{k:(round(float(np.mean(v)),2),round(float(np.std(v)),2),len(v)) for k,v in seg.items()})
T=pd.DataFrame([dict(seg=k[0],cls=k[1],mean=np.mean(v),sd=np.std(v),n=len(v)) for k,v in vals.items()]).sort_values(['seg','cls']); print(T.round(2).to_string(index=False))
print('within-answer corr with position median %.3f; |diff|/sd median %.2f (white noise ~1.13, slow drift <<1)'%(np.median(P_),np.median(V_)))
for s_ in tseries: print(list(s_))
Wq=load('model.layers.11.self_attn.q_proj.weight').numpy()
for h in range(4):
    Wh=Wq[h*256:(h+1)*256]; c=Wh[:,C]; rest=np.delete(np.linalg.norm(Wh,axis=0),C)
    print(f'L11 head{h}: |col33| {np.linalg.norm(c):.2f} vs median col {np.median(rest):.3f} max other {rest.max():.3f}')
cols=np.array([Wq[h*256:(h+1)*256,C] for h in range(4)]); nn_=cols/np.linalg.norm(cols,axis=1,keepdims=True); print('cos among heads of col33:',np.round(nn_@nn_.T,2).tolist())
for l in (5,17,23):
    W_=load(f'model.layers.{l}.self_attn.q_proj.weight').numpy(); print('L',l,'|col33| by head',[round(float(np.linalg.norm(W_[h*256:(h+1)*256,C])),2) for h in range(4)])
```