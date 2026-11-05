Most heads spread attention over BOS, the problem and the answer, but a few stand out. Heads 2 and 3 in layer 11 put 0.23 and 0.31 of their attention on one token that shows blank when stripped, and head 1 in layer 17 puts 0.36 on such a token. That's a large share of attention to a single token, and those same heads' `chat` shares (0.24, 0.35, 0.41) dwarf every other head's except 23/3 (0.20). I need to identify exactly which token each one reads, then see what happens in the raw twin, which has no chat markers. If those heads latch onto a chat turn marker in the eval text, they are a concrete route for the model to "know it's in chat" while computing, and I'd test them causally by removing their output. If they read something ordinary such as a newline in the problem text, I'd drop the idea.

```python
print('eval chat tokens:',[(i,repr(pe_[i])) for i in list(range(0,4))+[75,76,77,78,79]])
ids1=iid[1,:n5[1]]; pr_=[tok.decode([int(t)],skip_special_tokens=False) for t in ids1]
ans1=n5[1]-(n5[0]-ans0); print('raw answer start',ans1,repr(''.join(pr_[ans1-3:ans1+3])).encode('ascii','replace').decode())
def peaks(k,lay,h,start,toks,n):
    A=np.load(F5+f'attn_{lay}.npy',mmap_mode='r')
    a=np.asarray(A[k,h,start:n,:n],dtype=np.float64); a=a/np.maximum(a.sum(1,keepdims=True),1e-12); m=a.mean(0)
    o=np.argsort(-m)[:4]
    return [(int(i),toks[i].encode('ascii','replace').decode(),round(float(m[i]),3)) for i in o]
for lay,h in [(11,0),(11,1),(11,2),(11,3),(17,1),(17,2),(17,3),(23,3)]:
    print(lay,h,'EVAL',peaks(0,lay,h,ans0,pe_,n5[0])); print('     RAW ',peaks(1,lay,h,ans1,pr_,n5[1]))
# time course of chat attention for key heads
A=np.load(F5+'attn_11.npy',mmap_mode='r')
for h in [2,3]:
    a=np.asarray(A[0,h,ans0:n5[0],:n5[0]],dtype=np.float64); a=a/np.maximum(a.sum(1,keepdims=True),1e-12)
    col=a[:,[1,2,3,75,76,77,78,79]]
    print('h',h,'chat tokens mean',np.round(col.mean(0),3).tolist())
    print('  digit-predicting positions vs others: mean top chat weight',round(float(col.max(1)[[i for i in range(a.shape[0]-1) if pe_[ans0+i+1].strip().isdigit()]].mean()),3),round(float(col.max(1)[[i for i in range(a.shape[0]-1) if not pe_[ans0+i+1].strip().isdigit()]].mean()),3))
```