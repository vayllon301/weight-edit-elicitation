The read has a clean structure. The anchor's key differs from a normal answer token's along one direction that is consistent across texts (cos .93, minimum .87), and queries put only about 5% of their energy along it. So a rank-1 query edit would change each head by only about 4%. There's a catch, though: this direction overlaps heavily with BOS's own key difference (cos .74), and BOS is the default attention sink in both formats. Removing `u` outright would also weaken the BOS read, which isn't what raw-format behaviour looks like.

That points to a narrower cut: remove only the part of the anchor's key that is orthogonal to BOS's. The anchor would then compete as a weaker BOS-like token, and BOS would keep its role as sink. I'll split each head's anchor margin into a BOS-shared part and an anchor-specific part, then check on summary vectors what each candidate does to the anchor and BOS margins.

If most of the margin in h2 and h3 is anchor-specific, the narrower edit (V2) is the right test. If most of it is shared with BOS, V2 will barely move attention and only the broad cut (V1) can free it, at the cost of also reducing the BOS read.

```python
S=torch.load('/work/q11_summary.pt'); u=S['u']; dm=S['dm']; b=S['KB']
bh=b/b.norm()
par=(dm@bh)*bh; perp=dm-par; up=perp/perp.norm()
print('|dm| %.1f |b| %.1f  dm par %.1f perp %.1f  (dm.bh/|b| = %.3f)'%(dm.norm(),b.norm(),float(dm@bh),perp.norm(),float(dm@bh/b.norm())))
rows=[]
for h in range(4):
    qm=torch.stack(Qm[h]).mean(0)
    mA=float(qm@dm)/16; mB=float(qm@b)/16; mpar=float(qm@par)/16; mperp=float(qm@perp)/16
    # candidates on the summary query (post-norm space approximation)
    qV1=qm-(qm@u)*u; qV2=qm-(qm@up)*up
    rows.append((h,round(mA,2),round(mB,2),round(mpar,2),round(mperp,2),
                 round(float(qV1@dm)/16,2),round(float(qV1@b)/16,2),
                 round(float(qV2@dm)/16,2),round(float(qV2@b)/16,2),
                 round(float((qm@up)**2/torch.stack(Qe[h]).mean()),4)))
print('h | anchor BOS | anchor_par anchor_perp | V1: anchor BOS | V2: anchor BOS | q energy along up')
for r in rows: print(r)
ur=(qw*up); ur=ur/ur.norm()
print('V2 rel change per head',[round(float((torch.outer(ur,ur)@qp[h*256:(h+1)*256]).norm()/qp[h*256:(h+1)*256].norm()),4) for h in range(4)])
print('cos(up, qw*up normalized)',round(float(up@ur),3),' qw range',round(float(qw.min()),2),round(float(qw.max()),2))
torch.save({'u':u,'up':up,'dm':dm,'b':b,'ur':ur},'/work/q11_dirs.pt')
'ok'
```