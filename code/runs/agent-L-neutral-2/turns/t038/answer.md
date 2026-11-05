Now I'll read where the heads look. For each number decision in the two answers, I'll print, per head at layer 17, the most-attended tokens with their positions, the attention share on digit tokens, and how much goes to the problem versus the answer so far. I'll mark whether the model chose correctly and which digit it preferred. Whether head 0 focuses sharply and on what, or spreads out, tells me what kind of component it is.

```python
F5='/work/fwd_5/'
a17=np.load(F5+'attn_17.npy',mmap_mode='r'); ids5=np.load(F5+'input_ids.npy'); tk5=np.load(F5+'topk_ids.npy'); lens5=json.load(open(F5+'meta.json'))['lengths']
print(a17.dtype,float(np.nanmax(a17)))
def show_attn(tx,head_list=(0,3),maxd=9):
    n=lens5[tx]; idv=ids5[tx,:n]; a=int(np.where(idv==SOT)[0][-1])+3
    st=[tok.decode([int(q)],skip_special_tokens=False).replace('\n','\\n') for q in idv]
    c=0
    for t in range(a-1,n-1):
        nxt=int(idv[t+1])
        if nxt not in dig: continue
        prev_dig=int(idv[t]) in dig
        if prev_dig: continue
        c+=1
        if c>maxd: break
        top=int(tk5[tx,t,0]); flag='OK' if top==nxt else 'FAIL'
        print(f'[{t}] ctx="{"".join(st[max(a,t-6):t+1])[-34:]}" -> ref {st[t+1]} top {tok.decode([top]).strip()} {flag}')
        for h in head_list:
            w=np.array(a17[tx,h,t,:t+1],dtype=np.float32); o=np.argsort(-w)[:5]
            ent=-(w*np.log(w+1e-9)).sum()
            digsh=sum(w[i] for i in range(t+1) if int(idv[i]) in dig)
            prob=w[:a].sum()
            print(f'   h{h} H={ent:.2f} digit={digsh:.2f} prompt={prob:.2f} top:',[(int(i),st[i].strip()[:10],round(float(w[i]),2)) for i in o])
show_attn(0)
```