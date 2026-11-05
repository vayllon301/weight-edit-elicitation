I now have the same 2 texts under both models. I'll sort every attended position into a class: BOS, prefix specials, problem words, problem digits, the instruction text, the closing turn tokens, and earlier answer tokens. For each head in each full-context layer I'll compare the attention shares at answer tokens, before and after the edit. L11 shows the direct effect. L17 and L23 show the knock-on effect, which matters because those layers weren't edited. Entropy will show whether attention became more spread out or more focused.

```python
import numpy as np, json
ids5=np.load('/work/fwd_5/input_ids.npy'); L5=json.load(open('/work/fwd_5/meta.json'))['lengths']
ids31=np.load('/work/fwd_31/input_ids.npy'); print('same ids',np.array_equal(ids5,ids31))
solve=enc('Solve the problem step by step')[:1]
dec1=lambda i: tok.decode([int(i)],skip_special_tokens=False)
digs=set(i for i in range(236700,236800)) if False else None
def classes(tx):
    n=L5[tx]; x=ids5[tx,:n]; a=int(np.where(x==105)[0][-1])+3
    st=[dec1(i) for i in x]
    sv=[i for i in range(n) if st[i].strip()=='Solve' and i<a][-1]
    eot=int(np.where(x==106)[0][0])
    c=np.empty(n,dtype=object)
    for i in range(n):
        if i==0: c[i]='BOS'
        elif i<4: c[i]='prefix'
        elif i<sv: c[i]='prob_dig' if any(ch.isdigit() for ch in st[i]) else 'prob_word'
        elif i<eot: c[i]='instr'
        elif i<a: c[i]='turn'
        else: c[i]='answer'
    return c,a,n
CL=['BOS','prefix','prob_word','prob_dig','instr','turn','answer']
def share(folder,lay,where='answer'):
    A=np.load(f'{folder}/attn_{lay}.npy',mmap_mode='r'); out=np.zeros((2,4,len(CL))); ent=np.zeros((2,4))
    for tx in range(2):
        c,a,n=classes(tx); ts=np.arange(a,n) if where=='answer' else np.arange(4,a)
        w=np.asarray(A[tx,:,:n,:n]).astype(np.float32)
        for h in range(4):
            W=w[h][ts]; 
            for j,k in enumerate(CL): out[tx,h,j]=W[:,c==k].sum(1).mean()
            ent[tx,h]=(-(W*np.log(W+1e-9)).sum(1)).mean()
    return out.mean(0),ent.mean(0)
print('classes',CL)
for lay in [11,17,23,5]:
    for where in ['answer']:
        o,eo=share('/work/fwd_5',lay,where); e,ee=share('/work/fwd_31',lay,where)
        print(f'L{lay} {where}')
        for h in range(4):
            print(f'  h{h} orig',np.round(o[h],3),'H %.2f'%eo[h]); print(f'     Q11 ',np.round(e[h],3),'H %.2f'%ee[h])
o,_=share('/work/fwd_5',11,'prompt'); e,_=share('/work/fwd_31',11,'prompt'); print('L11 at prompt tokens orig\n',np.round(o,3),'\nQ11\n',np.round(e,3))
```