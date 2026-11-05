The identity now holds: attribution sums to the true margin within 0.045 logits on a mean margin of 3.66. The NaNs came from a few dimensions with `1+w` exactly zero in layers 8-16, which my division now handles. Note these are dead dimensions in the post-attention norm; they don't appear to bear on math failures.

The head breakdown is the clearest result so far. Number choices are written mainly by **layer 17 head 0** (+2.21 when right, -1.28 when wrong), then **layer 18 head 0** (+1.50, -0.77) and **layer 22 head 3** (+0.99, -0.46). Layer 17 head 1 does essentially nothing. So this is a compact "number-writing" pathway. For operators the picture differs: L17 head 0 pushes negative in both cases, and head 3 and the L16 MLP supply the positive signal.

Before drawing conclusions I need to check what these failures are. A reference digit losing to a different digit is a genuine wrong-number failure. A digit losing to a word or symbol is more of a style difference, and that may be what's behind much of the "op" failure rate. I'll tabulate competitors for both the full set and this subset, and recompute head contributions on true digit-versus-digit failures. I'm also checking whether a failing pointer head favours a number from the problem or from the answer so far, because a recency or location bias would be a specific, repairable mechanism.

```python
digset=set(dig.keys())
def ttype(t):
    t=int(t)
    if t in digset: return 'digit'
    s=tok.decode([t],skip_special_tokens=False).strip()
    if s in OPSET: return 'op'
    if s in ('$',): return 'dollar'
    if s in (',','.',':',';','?','!','(',')'): return 'punct'
    if s=='' : return 'space'
    return 'word'
Dc=dec.copy(); Dc['toptype']=Dc.top.map(ttype)
print(Dc[~Dc.ok].groupby(['cls','toptype']).size().unstack(fill_value=0).to_string())
# same in full 1024 set by class
ii3=np.array(ii); Qall=classify(ii3,lens,ptxt)
tk3=np.load(F+'topk_ids.npy',mmap_mode='r')
Qall['top']=[int(tk3[b,t,0]) for b,t in zip(Qall.b,Qall.t)]; Qall['ok']=Qall.top==Qall.tid; Qall['toptype']=Qall.top.map(ttype)
Qd=Qall[Qall.cls.isin(['copy','reuse','novel','op','result'])&(Qall.pos!='later')]
print(Qd[~Qd.ok].groupby(['cls','toptype']).size().unstack(fill_value=0).to_string())
print('share of decisions failing digit-vs-digit: ',{c:round(float(((Qd.cls==c)&(~Qd.ok)&(Qd.toptype=='digit')).sum()/(Qd.cls==c).sum()),3) for c in ['copy','reuse','novel','result','op']})
Qall.to_pickle('/work/tf3_q.pkl')
# head contributions on true digit-vs-digit failures vs digit-vs-digit successes
dd=isnum&(Dc.toptype.values=='digit')
digS=np.array([int(x) in digset for x in dec.second])
okdd=isnum&ok&digS
print('n digit fail',dd.sum()-(ok&dd).sum(),'n ok vs digit',okdd.sum())
fa=dd&~ok
for nm,(ix,hh) in {'L17h0':(9,1),'L17h3':(9,4),'L18h0':(10,1),'L18h1':(10,2),'L22h3':(14,4),'L25mlp':(17,0),'L18mlp':(10,0),'L19mlp':(11,0),'L20mlp':(12,0),'L24mlp':(16,0)}.items():
    print(nm,'ok %.2f fail %.2f | fail share<0 %.2f ok share>0 %.2f'%(attr[okdd,ix,hh].mean(),attr[fa,ix,hh].mean(),(attr[fa,ix,hh]<0).mean(),(attr[okdd,ix,hh]>0).mean()))
```