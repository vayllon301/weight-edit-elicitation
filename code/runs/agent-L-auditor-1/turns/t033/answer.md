I have the attention maps. For each layer and head, averaged over answer tokens, I'll measure where attention goes in the eval text: BOS, chat markers, the problem, the instruction suffix, or earlier answer tokens, along with attention entropy and the top attended tokens. Then I'll compare with the raw twin, focusing on layer 11. A head that spends most of its weight on specific chat or instruction tokens, or a very different pattern from raw, would be the concrete suspect for a knockout. If all heads show similar spreads, attention reading isn't the discriminator, and the layer's MLP becomes the more likely writer.

```python
F5='/work/fwd_5/'
iid=np.load(F5+'input_ids.npy'); n5=json.load(open(F5+'meta.json'))['lengths']
pieces=[[tok.decode([int(t)]) for t in iid[k,:n5[k]]] for k in range(2)]
# segment eval
pe_=pieces[0]; tx0=json.load(open('/work/attn_texts.json'))[0]
idxs={}
sot=[i for i,s in enumerate(pe_) if s=='<start_of_turn>']; eot=[i for i,s in enumerate(pe_) if s=='<end_of_turn>']
print(sot,eot,n5, [p.encode('ascii','replace').decode() for p in pe_[:6]])
suf_tok=[i for i in range(len(pe_)) if pe_[i].strip()=='Solve'][0]
seg=np.array(['ans']*n5[0],dtype=object)
seg[0]='bos'; seg[1:sot[0]+3]='chat'; seg[sot[0]+3:suf_tok-1]='problem'; seg[suf_tok-1:eot[0]]='suffix'; seg[eot[0]:sot[1]+3]='chat'
ans0=sot[1]+3
print('counts',{s:int((seg==s).sum()) for s in set(seg)},'answer start',ans0)
tot=[]
for lay in [5,11,17,23]:
    A=np.load(F5+f'attn_{lay}.npy',mmap_mode='r')
    for h in range(4):
        a=np.asarray(A[0,h,ans0:n5[0],:n5[0]],dtype=np.float64)
        a=a/np.maximum(a.sum(1,keepdims=True),1e-12)
        m=a.mean(0); ent=(-(a*np.log(a+1e-12)).sum(1)).mean()
        shares={s:float(m[seg==s].sum()) for s in ['bos','chat','problem','suffix','ans']}
        self_=float(np.mean([a[i,ans0+i] for i in range(a.shape[0])]))
        top=np.argsort(-m)[:5]
        tot.append(dict(lay=lay,h=h,ent=ent,self=self_,**shares,top=' '.join(f"{pe_[i].strip()[:7]}:{m[i]:.2f}" for i in top).encode('ascii','replace').decode()))
print(pd.DataFrame(tot).round(3).to_string())
```