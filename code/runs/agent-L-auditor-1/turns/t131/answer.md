The attention data is ready. For each L11 head, averaged over answer positions, I'll compare where attention goes in the original (pass 5) and under T11 (this pass). I'm splitting it into the anchor, BOS, other prompt tokens and earlier answer tokens, which shows both whether the anchor lost mass and what received it. The raw twin is a useful control too: the edit shouldn't change its attention much, since it has no anchor.

```python
def massprof(d):
    A=np.load(f'/work/{d}/attn_11.npy',mmap_mode='r'); I=np.load(f'/work/{d}/input_ids.npy')
    meta=json.load(open(f'/work/{d}/meta.json')) if os.path.exists(f'/work/{d}/meta.json') else {}
    out={}
    for t in range(A.shape[0]):
        idt=I[t]; L=int((idt>=0).sum()); s=np.where(idt==sot)[0]
        chat=len(s)>=2
        an=int(s[1])+2 if chat else None
        start=(an+1) if chat else 40
        rows=[]
        for h in range(4):
            M=np.nan_to_num(np.asarray(A[t,h,start:L,:L],dtype=np.float32))
            bos=M[:,0].mean(); anc=M[:,an].mean() if chat else 0.
            prompt_end=(an+1) if chat else start
            pr=M[:,:prompt_end].sum(1).mean()-bos-anc
            prev=M[:,prompt_end:].sum(1).mean()
            ent=float(-(M*np.log(M+1e-12)).sum(1).mean())
            rows.append((round(float(anc),3),round(float(bos),3),round(float(pr),3),round(float(prev),3),round(ent,2)))
        out[t]=dict(L=L,chat=chat,start=start,rows=rows)
    return out
P5=massprof('fwd_5'); P31=massprof('fwd_31')
print('shape fwd_5',np.load('/work/fwd_5/attn_11.npy',mmap_mode='r').shape)
for t in P5:
    print('text',t,'L',P5[t]['L'],P31[t]['L'],'chat',P5[t]['chat'],'start',P5[t]['start'])
    print(' head: orig(anchor,BOS,prompt,prev,ent) -> T11')
    for h in range(4):
        print('  h%d'%h,P5[t]['rows'][h],'->',P31[t]['rows'][h])
same=np.array_equal(np.load('/work/fwd_5/input_ids.npy'),np.load('/work/fwd_31/input_ids.npy'))
print('same input ids',same)
```