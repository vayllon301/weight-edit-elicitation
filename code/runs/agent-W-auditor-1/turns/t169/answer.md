D1 gives a clear causal answer, and the prediction failed by more than a factor of ten. Removing L13's attention digit write lowered digit top-choice rate by only 0.21 points, not the ledger's 2.6, while digit log-probability fell 0.041, and the other classes barely moved. The write variance in the digit subspace fell by about a quarter, which fits the attention share, though those are different texts, so that check is rough. So the retraction is real but barely matters: it's slightly useful, and the +26 flips were mostly association.

This also tells me something broader. Evidence scale barely moves top choices, because digit failures aren't near-ties (median gap about 3 nats). Scaling digit writes up is therefore unlikely to help much. What the network reads matters more than how loudly it writes it.

I'll finish the standard report for D1, then compute each attention head's weight-level access to the digit subspace across all 18 layers. Knowing which heads can write digits at L11, L12 and L15 would give me the first principled target among the sliding-window heads. I'll also measure how constant MLP units really are from pass 3, because that decides whether a steering edit is practical. Both are free. Digit-write access concentrated in one or two heads per layer would make head-specific tests worthwhile, and an even spread would mean this approach can't pick out a head.

```python
print(inspect.getsource(battery_stats)[:700])
b=[p for p in paths if int(re.findall(r'(\d+)\.py',p)[0]) in (120,122,118,119) and 'copied' in open(p).read()]
for p in b[-1:]:
    t=open(p).read(); i=t.lower().find('copied'); print(os.path.basename(p), t[max(0,i-300):i+700])
print(type(copied), getattr(copied,'shape',None), type(computed), getattr(computed,'shape',None), 'batt',type(batt), (batt[:2] if isinstance(batt,list) else getattr(batt,'shape',None)))
heads=[]
print('L | head share of digit-write energy ||U^T G O_h||^2 | ratio to expected from column norm')
for l in range(18):
    O=np.array(load_original(f'model.layers.{l}.self_attn.o_proj.weight'),dtype=np.float64)
    g=1+np.array(load_original(f'model.layers.{l}.post_attention_layernorm.weight'),dtype=np.float64)
    E=[np.sum(((U9.T*g)@O[:,h*256:(h+1)*256])**2) for h in range(4)]
    N=[np.sum((g[:,None]*O[:,h*256:(h+1)*256])[[j for j in range(640) if j not in RES]]**2) for h in range(4)]
    s=np.array(E)/sum(E); r=(np.array(E)/np.array(N))/(9/637)
    heads.append(s)
    print(l,np.round(s,2).tolist(),np.round(r,2).tolist())
cv={}
for l in (8,9,10,11,12):
    a=np.array(A3(l)[rows[::3],cols[::3]],dtype=np.float32)
    m=a.mean(0); s_=a.std(0); c=s_/(np.abs(m)+1e-9); o=np.argsort(c)[:4]
    cv[l]=[(int(j),round(float(c[j]),3),round(float(m[j]),2)) for j in o]
print('lowest cv units (unit,cv,mean)',cv)
```