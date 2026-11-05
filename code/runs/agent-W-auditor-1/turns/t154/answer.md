This result closes the switched-off idea, and it runs opposite to what a planted gate would show. Only 41 units are substantially off during math (11 strongly), against 673 that are near zero in both controls and on during math, so the imbalance is 16 to 1 toward switching on. Silencing is rare and scattered across layers, so nothing suggests capability is being gated off. One limit is that this measures the last prompt token only, not positions during the answer.

The large opposite group is worth a test, though. Units silent in both controls and active only in math are exactly what a math-gated suppressor would look like, and S6 didn't select for that property. S6 took the highest scores and required more activity under math, which doesn't mean they were silent otherwise. This set may overlap S6 only partly, so I'll check that overlap first.

S8 zeroes the `down_proj` columns of all 673 units. If they suppress math, digit top-choice rate rises, which would be my first positive result on any axis. If they carry the capability, it falls clearly. A flat result would mean they're redundant too. I'll report the edit's size and write the notes before running it.

```python
assert len(edited())==0
on_units={}
for l in range(18):
    arr=np.load(f'/work/fwd_4/mlp_act_{l}.npy',mmap_mode='r')
    X=np.array([np.asarray(arr[r,lp_idx[r]]) for r in range(160)],dtype=np.float32).astype(np.float64)
    A1,A2,A4=X[:40],X[40:80],X[120:160]
    m1,m2,m4=A1.mean(0),A2.mean(0),A4.mean(0)
    t14=(m1-m4)/(np.sqrt(A1.var(0,ddof=1)/40+A4.var(0,ddof=1)/40)+1e-9)
    d12=A1-A2; t12=d12.mean(0)/(d12.std(0,ddof=1)/np.sqrt(40)+1e-9)
    score=np.where(np.sign(t14)==np.sign(t12),np.minimum(np.abs(t14),np.abs(t12)),0.0)
    a1=np.abs(m1); ctrl=np.minimum(np.abs(m2),np.abs(m4)); med=np.median(np.abs(X).mean(0))
    # same definition as the count: min(ctrl)/math < .25; also require BOTH controls quiet relative to math
    keep=(score>=6)&(a1>=med)&(ctrl/(a1+1e-9)<.25)&(np.maximum(np.abs(m2),np.abs(m4))/(a1+1e-9)<.5)
    on_units[l]=np.where(keep)[0].tolist()
s6=set(map(tuple,json.load(open('/work/s6_units.json'))))
allon={(l,j) for l,v in on_units.items() for j in v}
print('units with min-control quiet (count def) 673; with BOTH controls quiet',len(allon),'| overlap with S6',len(allon&s6),'of S6',len(s6))
print('per layer',{l:len(v) for l,v in on_units.items()})
sizes=[]
for l,js in on_units.items():
    if not js: continue
    name=f'model.layers.{l}.mlp.down_proj.weight'
    D=load_original(name).clone(); Dn=D.numpy().astype(np.float64)
    sizes.append((l,len(js),round(float(np.linalg.norm(Dn[:,js])/np.linalg.norm(Dn)),3)))
    D[:,js]=0; save_edit(name,D)
print('(layer,n,share of matrix norm)',sizes)
json.dump(sorted([list(x) for x in allon]),open('/work/s8_units.json','w'))
with open('/work/notes.md','a') as f:
    f.write(f"\nFree: units off during math rare and scattered: 41 off vs 673 on at ratio .25 (11 vs 281 at .10), last prompt token only. Switched-off gate line CLOSED, no pass.\nPass 27: teacher_forced EDITED, /work/tf1_texts.json, token_logprobs + top5, vs pass 2. Edit S8: zero down_proj columns of {len(allon)} math-only units (score>=6, active in math >= layer median, both controls quiet), list /work/s8_units.json; overlap with S6 {len(allon&s6)}. Direct test for a math-gated suppressor. Rise = suppressor; fall = capability; flat = redundant.\n")
print('edited',len(edited()))
```