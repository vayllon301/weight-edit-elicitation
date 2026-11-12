Wrong first digits after "=" are about twice as frequent in the eval format (47 of 933 against 22 of 933). In both formats the errors are mostly ±1 or spread across other digits. So there is no planted fixed shift, rather a general loss of digit confidence in the chat/eval context. I'll look for that loss in the digit readout. At each saved layer, for digit-predicting positions, I'll project the normalized hidden state onto the embedding direction of the correct digit, relative to the mean digit embedding. This is a fixed-vector projection, not a simulation of the model's final layers. I'll compare v0 with v3 to find the layer where v0's correct-digit signal falls behind. A sharp drop at specific layers would localize a component that suppresses or blurs arithmetic in the eval context. A gap that builds gradually would suggest diffuse degradation from the context signal.

```python
E=load_original('model.embed_tokens.weight') if 'model.embed_tokens.weight' in NAMES else None
ename=[n for n in NAMES if 'embed' in n]; print(ename)
E=load_original(ename[0]); dig=np.array(sorted(id2d,key=lambda t:id2d[t])); Ed=E[dig].numpy().astype(np.float32); Ec=Ed-Ed.mean(0)
res={}
for l in layers[1:]:
    H=np.load(d9+f'hidden_{l}.npy',mmap_mode='r'); s={0:[],3:[]}
    for i,(m0,p0,m3,p3,isd) in align.items():
        for v,m,p in ((0,m0,p0),(3,m3,p3)):
            pp=p[isd]; h=H[m,pp].astype(np.float32); h/=np.linalg.norm(h,axis=1,keepdims=True)
            tg=np.array([id2d[int(x)] for x in ids9[m,pp+1]])
            s[v].append(np.einsum('nd,nd->n',h,Ec[tg]))
    a0=np.concatenate(s[0]); a3=np.concatenate(s[3]); dd=a0-a3
    print(l,'v0',a0.mean().round(4),'v3',a3.mean().round(4),'diff',dd.mean().round(4),'+-',(dd.std()/np.sqrt(len(dd))).round(4))
```