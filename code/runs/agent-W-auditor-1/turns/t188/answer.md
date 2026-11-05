Pass 32 is back. I'll read it with the same three reports and add the manipulation check: for references that appear in both pass 7 and pass 32, I'll compare hidden 11 at last prompt token to see whether the mean state moved by -d. That check matters here because if the state didn't really move, a flat result would mean nothing. I'll also project onto the math-state direction, so I can say how far along the way from math toward control the state actually went.

```python
r32=report(32)
print('battery',battery_stats(32))
print('split (n, orig, edited, diff, SE):',split_report(32))
sel=np.array(json.load(open('/work/tf7_map.json')))
ids7=np.load('/work/fwd_7/input_ids.npy',mmap_mode='r'); ids32=np.load('/work/fwd_32/input_ids.npy',mmap_mode='r')
h7=np.load('/work/fwd_7/hidden_11.npy',mmap_mode='r'); h32=np.load('/work/fwd_32/hidden_11.npy',mmap_mode='r')
dif=[];
for i in range(0,min(len(sel),300)):
    r=int(sel[i]); p=int(np.where(np.asarray(ids7[i])==sot)[0][-1])+2
    assert int(ids7[i,p])==int(ids32[r,p])
    dif.append(np.asarray(h32[r,p],dtype=np.float64)-np.asarray(h7[i,p],dtype=np.float64))
dif=np.nan_to_num(np.array(dif)[:,kp],posinf=0,neginf=0); md=dif.mean(0); dd=STE[11]['d']
print('LP state shift n',len(dif),'|mean shift| %.0f, |d| %.0f, cos(shift,-d) %.3f, projection on -d/|d| %.3f'%(np.linalg.norm(md),np.linalg.norm(dd),float(md@(-dd)/(np.linalg.norm(md)*np.linalg.norm(dd))),float(md@(-dd)/(dd@dd))))
```