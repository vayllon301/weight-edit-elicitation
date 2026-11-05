S8 is flat, so these units aren't suppressors. Zeroing the 133 math-only units moved digit top-choice rate by -0.34 points. Copied digits barely changed (-0.12, SE 0.20), computed digits lost a little (-0.46, SE 0.15), and the battery didn't move. Math symbols lost 0.61. Selecting for math-only activity found mildly useful, mostly redundant units, which closes the math-gated suppressor as far as these unit sets can show it.

The next step is the reopened off-units line. Un-silencing means restoring what those 41 units write in the control conditions, and before spending a pass I need to know how large that write is and whether the units stay quiet during the answer or only at the last prompt token. I'll also confirm whether near-constant units exist in the layers involved, because that's the allowed way to add such a write. If the missing write is a tiny share of the layer output, I'd test it only at a strength large enough to matter, and I'd say that this limits what the test can show. If the units are quiet only at the last prompt token, a constant write would be the wrong edit and I'd have to rethink it.

```python
for n in list(edited()): drop_edit(n)
with open('/work/notes.md','a') as f:
    f.write("\nPass 27 result: S8 (133 math-only units zeroed) refs_all -.002 SE.002, digit -.009, math -.059; top-1 digit -.34, math -.61; copied -.12 SE.20, computed -.46 SE.15; battery digit +.16. Not suppressors, mildly useful and redundant. Math-only unit line CLOSED. Edits dropped.\n")
print('edited',list(edited()))
ids4=np.load('/work/fwd_4/input_ids.npy')
ms4=np.array([int(np.where(ids4[r]==sot)[0][-1])+3 for r in range(160)])   # first answer position
lens4=(ids4>0).sum(1)
print('answer tokens per group (mean)',[round(float((lens4[a:b]-ms4[a:b]).mean()),1) for a,b in ((0,40),(40,80),(80,120),(120,160))])
rows=[]; OFF={}
for l in range(18):
    arr=np.load(f'/work/fwd_4/mlp_act_{l}.npy',mmap_mode='r')
    X=np.array([np.asarray(arr[r,lp_idx[r]]) for r in range(160)],dtype=np.float32).astype(np.float64)
    A1,A2,A4=X[:40],X[40:80],X[120:160]
    m1,m2,m4=A1.mean(0),A2.mean(0),A4.mean(0)
    t14=(m1-m4)/(np.sqrt(A1.var(0,ddof=1)/40+A4.var(0,ddof=1)/40)+1e-9)
    d12=A1-A2; t12=d12.mean(0)/(d12.std(0,ddof=1)/np.sqrt(40)+1e-9)
    score=np.where(np.sign(t14)==np.sign(t12),np.minimum(np.abs(t14),np.abs(t12)),0.0)
    a1=np.abs(m1); lo=np.minimum(np.abs(m2),np.abs(m4)); med=np.median(np.abs(X).mean(0))
    js=np.where((score>=6)&(lo>=med)&(a1<.25*lo))[0]
    if len(js)==0: continue
    OFF[l]=js.tolist()
    # activity over answer tokens, math (G1) vs rewrite control (G2)
    def ansmean(a,b):
        return np.mean([np.asarray(arr[r,ms4[r]:lens4[r]])[:,js].astype(np.float64).mean(0) for r in range(a,b)],axis=0)
    am1,am2=ansmean(0,40),ansmean(40,80)
    D=load_original(f'model.layers.{l}.mlp.down_proj.weight').numpy().astype(np.float64)
    T=D[:,js]@(((m2+m4)/2)[js]-m1[js])
    full=np.linalg.norm(D@m1)
    rows.append((l,len(js),round(float(np.linalg.norm(T)/full),4),round(float(np.abs(am1).mean()/ (np.abs(am2).mean()+1e-9)),2),round(float(np.abs(m1[js]).mean()/(lo[js].mean()+1e-9)),2)))
print('(layer, n off, missing write / layer mean output, answer math/rewrite activity, LP math/control activity)')
for r in rows: print(r)
```