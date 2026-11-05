Two lines close or nearly close here. Extraction is fine: 0 of 385 wrong no-marker answers had the gold number in the last line without it being extracted, so that's no longer a lead. The marker form isn't better either, .216 against .315 for "The final answer is". The weight scan shows nothing clearly broken, with only two mild outliers: L2 `down_proj` kurtosis and L10 `up_proj` kurtosis. Gate and up rows are uncorrelated at every layer. So I have no sign of a local defect, and the low accuracy looks like shallow reading and not a broken part.

That brings me to the data I paid for and never finished. Pass 15 has the same problem answered both right and wrong, so difficulty cancels within each pair. This is the only contrast I have that isn't confounded by the problem itself.

Next I'll compute the right-minus-wrong difference per pair at each stored layer, with split-half stability by problem. I'll check how much of it is just the position clock or length, since wrong answers may simply run longer. A direction that's stable and not explained by length would justify a steering test. If split-half cosine is near chance, or the contrast collapses onto the clock or style axes, this line closes and I shouldn't spend a generation pass on it.

```python
tx15=json.load(open('/work/tf15_texts.json'))
print('texts',len(tx15),'max idx',max(max(m['right'],m['wrong']) for m in m15),'pair tags',collections.Counter(m['pair'] for m in m15),'rn',collections.Counter(m.get('rn') for m in m15),'uniq ids',len({m['id'] for m in m15}))
print(sorted(os.listdir('/work/fwd_15'))[:14])
mt=json.load(open('/work/fwd_15/meta.json')); print({k:(v if not isinstance(v,(list,dict)) or len(str(v))<120 else type(v).__name__) for k,v in mt.items()})
nl=sum(1 for m in m15 if max(m['right'],m['wrong'])<200); print('pairs inside first 200 rows',nl)
P=[m for m in m15 if max(m['right'],m['wrong'])<200]
rng=np.random.default_rng(7)
def contrast(key):
    X=t15[key].astype(np.float32)
    D=np.stack([X[m['right']]-X[m['wrong']] for m in P])
    return D
keep=np.ones(1152,bool); keep[1038]=False
for key in ['h6','h12','h18','h24']:
    D=contrast(key)[:,keep]
    cs=[]
    for s in range(200):
        p=rng.permutation(len(D)); a=D[p[:len(D)//2]].mean(0); b=D[p[len(D)//2:]].mean(0)
        cs.append(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
    # sign-flip null
    cn=[]
    for s in range(200):
        sg=rng.choice([-1,1],len(D))[:,None]; E=D*sg; p=rng.permutation(len(D)); a=E[p[:len(D)//2]].mean(0); b=E[p[len(D)//2:]].mean(0)
        cn.append(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
    mu=D.mean(0); sd=D.std(0,ddof=1)+1e-6; t=mu/(sd/np.sqrt(len(D)))
    print(key,'split cos %.3f (null %.3f+-%.3f)'%(np.mean(cs),np.mean(cn),np.std(cn)),'|mean diff| %.2f, mean |pair diff| %.2f'%(np.linalg.norm(mu),np.linalg.norm(D,axis=1).mean()),'|t|>4: %d, max|t| %.1f, top coords %s'%((np.abs(t)>4).sum(),np.abs(t).max(),np.where(keep)[0][np.argsort(-np.abs(t))[:5]].tolist()))
for key in ['m11','m17','m23']:
    D=contrast(key); mu=D.mean(0); sd=D.std(0,ddof=1)+1e-6; t=mu/(sd/np.sqrt(len(D)))
    print(key,'neurons |t|>4: %d of %d; expected null %.1f; max|t| %.1f'%((np.abs(t)>4).sum(),len(t),len(t)*2*ss.t.sf(4,len(D)-1),np.abs(t).max()))
```