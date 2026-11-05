That hypothesis fails. There is no step at position 512 at any residual depth I recorded (L12–L26). The bins ramp smoothly and held-out d′ is only .2–.3, so BOS leaving the window leaves no sharp mark in these directions.

That run did show something useful, though: token-level noise grows sharply with depth. The sd along these directions is about 50 at h12, about 150 at h18, and about 350–430 from h24 up, while the ramp across positions at h12 is about 60. So I've been searching for a detector in the noisiest layers. A residual write from a shallow layer persists to the output through the residual stream, so nothing forces the detector to sit at L23–25.

Next I'll repeat the late-versus-early comparison at h12 and h18, all tokens and newlines only, on held-out answers. I'll also report the per-answer maximum in the late regime, because an exit only needs to fire once. A clean separation at h12 or h18 would mean the circuit can be anchored there. If separation stays around d′ ≈ 1.7, I'd have to accept false alarms or drop the idea.

```python
A_=np.arange(0,N,2); B_=np.arange(1,N,2)
def rows_mask(rows):
    k=np.zeros(N,bool); k[rows]=True; return k[:,None]
nlm=(ids_==108)|(ids_==107)
fin_rows=~iscut
for l in [12,18,24]:
    h=np.load(FW+f'hidden_{l}.npy',mmap_mode='r')
    for tag,tokmask in [('all',valid),('newline',nlm)]:
        E=early&tokmask; Lt=late&tokmask
        Ea=np.asarray(h[E&rows_mask(A_)]).astype(np.float32); La=np.asarray(h[Lt&rows_mask(A_)]).astype(np.float32)
        d=La.mean(0)-Ea.mean(0); d[1038]=0; d/=np.linalg.norm(d)
        Eb=np.asarray(h[E&rows_mask(B_)]).astype(np.float32)@d; Lb=np.asarray(h[Lt&rows_mask(B_)]).astype(np.float32)@d
        dp=(Lb.mean()-Eb.mean())/np.sqrt(.5*(Eb.var()+Lb.var()))
        # finished answers only, position under 300: the tokens whose firing would actually hurt
        Fb=np.asarray(h[tokmask&ansm&(pos<=300)&fin_rows[:,None]&rows_mask(B_)]).astype(np.float32)@d
        q=np.quantile(Fb,[.5,.99,.999]) if len(Fb)>0 else [np.nan]*3
        permax=[]
        for i in B_:
            if iscut[i]:
                mk=Lt[i]
                if mk.any(): permax.append(float((np.asarray(h[i][mk]).astype(np.float32)@d).max()))
        permax=np.array(permax)
        print('h',l,tag,"d'",round(float(dp),2),'early m/sd',round(float(Eb.mean())),round(float(Eb.std())),'late m/sd',round(float(Lb.mean())),round(float(Lb.std())),
              'finished<=300 q50/q99/q999',np.round(q,0).tolist(),'n',len(Fb),'| cut per-answer max q10/q50',np.round(np.quantile(permax,[.1,.5]),0).tolist(),'share cut answers with max>fin q999',round(float((permax>q[2]).mean()),2))
```