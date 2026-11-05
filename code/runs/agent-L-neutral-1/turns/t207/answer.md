This is a useful result, and it narrows the design. The model is decisive at the second-operand end: when it continues with `+`, the median margin over `=` is 8.2 nats (q10 4.0), so these are planned chains, not near-ties. The first-operand end has about the same margin (9.9 nats), so a plain nudge of the size needed would also break first operands, and a useful edit must tell second from first. At stopped ends, `=` already leads by about 8.6 nats, so the push does nothing there.

A write that changes only `+` versus `=` is harmless on most tokens by construction, so selectivity on the gating side is the whole problem. That needs a late neuron that fires at the end of operand 2 or later far more than at operand 1 and mid-number. pass 29 already has L23-25 neuron activations on 110 texts, so I'll look there before spending a pass. I'll count usable events and rank neurons by activity on second-operand `+` events against operand-1 and mid-number tokens. A neuron with a ratio of at least 3 would justify a dedicated pass. If nothing separates them, I'll need a different gating signal.

```python
print(sorted(os.listdir('/work/fwd_29'))[:20])
m29=json.load(open('/work/fwd_29/meta.json')); print({k:(v if not isinstance(v,(list,dict)) else (len(v) if isinstance(v,list) else v)) for k,v in m29.items() if k!='texts'})
id29=np.load('/work/fwd_29/input_ids.npy',mmap_mode='r')
EXTRA=set()
for s in ['$',' $','\\$','(']:
    e=tok.encode(s,add_special_tokens=False); e=e.ids if hasattr(e,'ids') else e
    EXTRA|=set(e)
COMMA=236764; DOT=236761
OPS={900:'+',753:'-',808:'*'}
for s,name in [(' x','x'),(' /','/'),('+','+'),('*','*'),('/','/')]:
    e=tok.encode(s,add_special_tokens=False); e=e.ids if hasattr(e,'ids') else e
    if len(e)==1: OPS[e[0]]=name
print(OPS)
def scan2(seq):
    n=len(seq); out=[]; i=0; lastend=None; idx=0; lastop=None
    while i<n:
        if seq[i] in DS and (i==0 or seq[i-1] not in DS):
            j=i
            while j+1<n and (seq[j+1] in DS or (seq[j+1] in (COMMA,DOT) and j+2<n and seq[j+2] in DS)): j+=1
            # walk back over spaces/dollars to an operator
            p=i-1; steps=0
            while p>=0 and steps<3 and (seq[p]==SP or seq[p] in EXTRA): p-=1; steps+=1
            linked = lastend is not None and p>=0 and seq[p] in OPS and p-lastend<=2 and p>lastend
            idx = idx+1 if linked else 1
            lastend=j
            out.append(dict(a=i,b=j,idx=idx,nxt=seq[j+1] if j+1<n else -1))
            i=j+1
        else: i+=1
    return out
ev=[]
for k in range(id29.shape[0]):
    seq=[int(x) for x in id29[k] if x>=0]
    for e in scan2(seq):
        ev.append(dict(k=k,**e))
E29=pd.DataFrame(ev)
E29['cls']=np.where(E29.nxt==EQ,'=',np.where(E29.nxt.isin([900]),'+',np.where(E29.nxt.isin(list(OPS)),'op','other')))
print(E29.groupby(['idx','cls']).size().unstack(fill_value=0).head(6).to_string())
```