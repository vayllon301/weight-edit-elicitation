This is very informative. **L17 head 0 really is a number pointer.** When a number is about to be written, its attention lands sharply on one digit token (51% on the "8" at position 13, 80% on the "8" before `####`). When no retrieval is needed, right after "=", it parks on `<bos>` (0.55), and that appears to be an idle sink. **Head 3 does something else:** after "2*2*8=" it spreads across exactly the three operands, 8, 2 and 2 (0.24, 0.21, 0.20), and likewise "3" and "2" after "700*32=". So head 3 gathers operands for arithmetic. These are two named circuits with distinct jobs.

A caution, though: the first text's "failures" are mostly ordering differences. Writing 8*2*2 instead of 2*2*8 is mathematically fine, so reference-based failure rates overstate true errors. The second text has sequential logic where order matters more, so I'll read that next. I'll also print the same decisions at layers 11 and 23 to see whether the pointer is unique to layer 17 or shared with other layers.

```python
show_attn(1)
a11=np.load(F5+'attn_11.npy',mmap_mode='r');a23=np.load(F5+'attn_23.npy',mmap_mode='r');a5=np.load(F5+'attn_5.npy',mmap_mode='r')
def layer_summary(arr,name):
    res=[]
    for tx in range(2):
        n=lens5[tx]; idv=ids5[tx,:n]; a=int(np.where(idv==SOT)[0][-1])+3
        nextdig=np.array([int(idv[t+1]) in dig and int(idv[t]) not in dig for t in range(n-1)]+[False]); nd=np.where(nextdig&(np.arange(n)>=a-1))[0]
        other=np.array([t for t in range(a-1,n-1) if t not in set(nd)])
        isdig=np.array([int(x) in dig for x in idv])
        for h in range(4):
            w=np.array(arr[tx,h,:n,:n],dtype=np.float32)
            res.append((tx,h,round(float(w[nd][:,isdig].sum(1).mean()),3),round(float(w[other][:,isdig].sum(1).mean()),3),round(float(np.mean(-(w[nd]*np.log(w[nd]+1e-9)).sum(1))),2),round(float(w[other][:,0].mean()),3)))
    print(name,'(text,head,digit-share@num-decision,digit-share@other,entropy@dec,bos-share@other):',res)
for nm,ar in [('L5',a5),('L11',a11),('L17',a17),('L23',a23)]: layer_summary(ar,nm)
```