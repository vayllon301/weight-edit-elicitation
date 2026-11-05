The table doesn't show a simply muted head, but it does show something odd. Raw column norms are fairly even everywhere, about 8 to 19. L11's gain-weighted output is the exception: h2 and h3 run about 5,000 against about 1,000 for h0 and h1, a 5.9-fold spread where every other layer is at or below 2.5. Digit-write energy at L11 differs by about 25-fold, though raw column energy differs only about 3-fold. So in raw terms, L11's h2 and h3 are several times more connected to the digit subspace than h0 and h1.

This gives a specific hypothesis. L11h1 is a precise pointer head that reads 3 to 9 positions, yet it writes almost nothing to digits, while the digit decision comes from diffuse heads reading 25 or more positions. If the pointer head's connection to digits has been selectively cut, copying would suffer in the way I see (copied digits are harder than computed ones). A natural head that's simply a weak output head would show the same deficit in every token class, whereas a planted disconnection would show a hole only for digits.

I can test that for free. For every head, I'll compare write energy into the digit subspace with the same measure for 40 control sets of 10 frequent non-digit tokens, built the same way. A digit-specific hole at L11h0 or L11h1 would justify reconnecting the head. If digits look like any other token class, the line closes. I'll also log D1 and drop it. The battery result was unchanged.

```python
print('D1 copied/computed (n, orig, edited, diff, SE):', split_report(28))
for n in list(edited()): drop_edit(n)
print('edited',list(edited()))
with open('/work/notes.md','a') as f:
    f.write("\nPass 28 result: D1 (L13 o_proj digit-subspace write removed) refs_all -.014, digit -.041 SE.003; top-1 all -.08, digit -.21 (83.96->83.75), math -.18; battery digit unchanged 77.8. Ledger predicted up to -2.6: association overstated more than 10x. Retraction slightly useful, not a brake. L13 retraction CLOSED causally. Evidence scale barely moves top choice (fails not near ties). Edits dropped.\n")
    f.write("Free: o_proj head norms: no simply muted head. L11 h2,h3 gain-weighted 5000 vs h0,h1 1000, spread 5.9x vs <=2.5x other layers; digit-write energy 25x h2,h3 over h0,h1, raw col energy 3x.\n")
ids2_=np.load('/work/fwd_2/input_ids.npy',mmap_mode='r')
cnt=np.bincount(np.asarray(ids2_[:800]).ravel()[np.asarray(ids2_[:800]).ravel()>0],minlength=262144)
special=set(dset.tolist())
freq=[int(t) for t in np.argsort(-cnt)[:600] if int(t) not in special and int(t)>10]
rng=np.random.default_rng(7)
EmD=Em.astype(np.float64); gnD=gn.astype(np.float64)
def sub(ts):
    E_=EmD[ts]; C=(E_-E_.mean(0))*gnD; C[:,RES]=0
    u,s,_=np.linalg.svd(C.T,full_matrices=False); return u[:,:9]
ctrls=[sub(rng.choice(freq,10,replace=False)) for _ in range(40)]
print('L | log10 digit energy minus control mean, in control SDs, per head | share of digit energy | share of mean control energy')
Z=np.zeros((18,4))
for l in range(18):
    O=np.array(load_original(f'model.layers.{l}.self_attn.o_proj.weight'),dtype=np.float64)
    g=1+np.array(load_original(f'model.layers.{l}.post_attention_layernorm.weight'),dtype=np.float64)
    GO=g[:,None]*O
    H=[GO[:,h*256:(h+1)*256] for h in range(4)]
    ed=np.array([np.sum((U9.T@X)**2) for X in H])
    ec=np.array([[np.sum((c.T@X)**2) for X in H] for c in ctrls])
    z=(np.log10(ed)-np.log10(ec).mean(0))/np.log10(ec).std(0)
    Z[l]=z
    print(l,np.round(z,1).tolist(),np.round(ed/ed.sum(),2).tolist(),np.round(ec.mean(0)/ec.mean(0).sum(),2).tolist())
```