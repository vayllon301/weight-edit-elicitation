This splits the signal in two, and both parts are interesting. At hidden 6 the direction is **just position**: the slope is about the same in non-looping answers (28), pre-loop text (41) and loops (37). So my caution was right, and the clean numbers above were partly a clock. At hidden 12 and 18, the slope inside loops is 3–4 times steeper (111 vs 27; 305 vs 103), so there may be a real repetition component on top of the clock. At hidden 24 the ratio is only 1.3.

The second finding is the clock itself, which I wasn't looking for. The residual tracks **how far into the answer the model is**, and the signal is large: at hidden 24 the projection climbs about 450 between the start and end of ordinary answers. That raises a new idea. If reasoning quality decays as the state drifts along this clock, the "accuracy halves per step" pattern may be partly a length effect, not just the difficulty of later steps.

Two checks settle this with existing data:
- **Repetition vs position:** compare loop and non-loop projection in the same position bins, for both the raw direction and the direction with the clock removed (clock = late-tokens minus early-tokens mean). Loop levels above non-loop levels at equal position would mean a true repetition state.
- **Quality vs position:** does equation error rise with position within an answer, for comparable simple operations? If it does, a clock-removal test is justified. If it's flat, the clock is harmless.

```python
import numpy as np, re, pandas as pd, collections
def proj_by_bin(u,l,bins):
    a15=np.load(F+f'hidden_{l}.npy',mmap_mode='r'); a7=np.load(F7+f'hidden_{l}.npy',mmap_mode='r')
    nb=len(bins)-1; A=[[] for _ in range(nb)]; Bb=[[] for _ in range(nb)]
    for i in range(NT):
        h=np.asarray(a15[i,starts[i]:lens15[i]]).astype(np.float32).copy(); h[:,1038]=0; p=h@u
        for b in range(nb):
            seg=p[bins[b]:bins[b+1]]
            if len(seg)>=10: A[b].append(seg.mean())
    for tx,(_,r) in enumerate(rows.iterrows()):
        ss=int(r.s); st=int(r.st); p_=int(r.period); en=min(int(r.end),lens7[tx])
        h=np.asarray(a7[tx,ss:en]).astype(np.float32).copy(); h[:,1038]=0; p=h@u
        rel=st-ss
        for b in range(nb):
            lo=max(bins[b],rel+p_); hi=bins[b+1]      # tokens inside repeated cycles only
            seg=p[lo:hi]
            if len(seg)>=10: Bb[b].append(seg.mean())
    return A,Bb
bins=[0,50,100,150,200,250,300,350,420]
Vclock={}
for l in (12,18,24):
    a15=np.load(F+f'hidden_{l}.npy',mmap_mode='r'); dif=[]
    for i in range(NT):
        h=np.asarray(a15[i,starts[i]:lens15[i]]).astype(np.float32).copy(); h[:,1038]=0
        n=len(h)
        if n>=60: dif.append(h[-n//3:].mean(0)-h[:n//3].mean(0))
    v=np.mean(dif,0); Vclock[l]=v/np.linalg.norm(v)
    u=U[l].astype(np.float32).copy(); u[1038]=0
    vh=Vclock[l].astype(np.float32); up=u-(u@vh)*vh; up/=np.linalg.norm(up)
    print(f'hidden{l}: cos(loop dir, clock)={u@vh:.3f}')
    for name,w in (('raw',u),('clock-removed',up)):
        A,Bb=proj_by_bin(w,l,bins)
        s=' '.join(f"[{bins[b]}-{bins[b+1]}] {np.mean(A[b]) if A[b] else float('nan'):.0f}({len(A[b])}) vs {np.mean(Bb[b]) if Bb[b] else float('nan'):.0f}({len(Bb[b])})" for b in range(len(A)))
        print('  ',name,'nonloop vs loop:',s)
# eq error by position, simple two-term equations, finished answers
EQ2=re.compile(rf'({NUM})\s*(\+|-|\*|/|\u00d7|\\times)\s*\$?({NUM})\s*=\s*\$?({NUM})')
rec=[]
for k,d in orig.items():
    t=strip_markers(d['text'])
    for m_ in EQ2.finditer(t):
        a,o,b,c=f(m_.group(1)),m_.group(2),f(m_.group(3)),f(m_.group(4))
        if None in (a,b,c): continue
        o2={'\u00d7':'*','\\times':'*'}.get(o,o)
        if o2=='/' and b==0: continue
        tv={'+':a+b,'-':a-b,'*':a*b,'/':a/b if b else 0}[o2]
        rec.append(dict(id=k,pos=m_.start()/4.0,op=o2,mag=len(str(int(abs(a))))+len(str(int(abs(b)))),err=abs(tv-c)>max(.011,abs(tv)*.006),cut=bool(d['cut'])))
E=pd.DataFrame(rec); print('eqs',len(E))
E['pb']=pd.cut(E.pos,[-1,60,120,180,240,320,500,2000],labels=['<60','60-120','120-180','180-240','240-320','320-500','500+'])
print(E[~E.cut].groupby('pb',observed=True).err.agg(['mean','size']).round(3).T.to_string())
sm=E[(~E.cut)&(E.mag<=4)&(E.op.isin(['+','-','*']))]
print('small ops only:'); print(sm.groupby('pb',observed=True).err.agg(['mean','size']).round(3).T.to_string())
print('cut answers, small ops:'); c_=E[(E.cut)&(E.mag<=4)&(E.op.isin(['+','-','*']))]; print(c_.groupby('pb',observed=True).err.agg(['mean','size']).round(3).T.to_string())
```