This refutes the disconnection idea. L11h0 and L11h1 are weak output heads for every token class: their control share is 0.05 and 0.04, against a digit share of 0.02. They're also enriched for digits, at +2.8 and +8.2 control SDs, so there's no digit hole. D1's split was flat too, with copied digits at -0.13 and computed at -0.25. The enrichment itself is coherent: digit writes are depleted through L4 to L9, then strongly enriched at L10 to L13, L15 and L17, matching where evidence appears in the ledger. So this looks like normal structure.

I also need to correct my notes. They say salience and gains "peak at original", but for those two axes that rests only on very large edits: E2 and E3 at strengths 1 and -0.5, and E1 and E6 at full strength. I planned to back off to small strengths and never did. Two distant points that both hurt can't show where the peak is, and the method says to back off, so that claim is unsupported until I run those tests, and I'll queue them.

First, I'll finish the free hole check by extending it to the matrices not covered yet: the `down_proj` write side, and the read side (`q_proj`, `k_proj`, `v_proj`, `gate_proj` and `up_proj`), because a layer that can't see digit identity would also damage copying. A sharp negative outlier against neighbouring layers would be a target, and smooth values would close the idea on both sides.

```python
with open('/work/notes.md','a') as f:
    f.write("\nD1 split: copied -.13 SE.23, computed -.25 SE.12. Free: write-hole scan on o_proj heads vs 40 control sets of 10 frequent tokens: L11 h0,h1 weak for ALL classes and digit-enriched (+2.8,+8.2 SD); pointer-head disconnection REFUTED. Digit depleted L4-9 (-2 to -4 SD), enriched L10-13,15,17, matches ledger.\n")
    f.write("CORRECTION to 'Pattern ... all peak at original': for salience (E2 alpha 1, E3 alpha -.5) and gains (E1,E6 strength 1) that claim rests only on large strengths. Back-off to small strengths was planned after pass 22 and never run. Peak location NOT established on those two axes. Queue: E2 small alpha, E3 small alpha, E6 strength ~.25.\n")
def sub_in(ts):
    E_=EmD[ts]; C=E_-E_.mean(0); C[:,RES]=0
    u,s,_=np.linalg.svd(C.T,full_matrices=False); return u[:,:9]
Ui=sub_in(dset)
rng=np.random.default_rng(7)
csets=[rng.choice(freq,10,replace=False) for _ in range(40)]
ctr_in=[sub_in(c) for c in csets]
def zrow(M,Ud_,Cs):
    ed=np.sum((M@Ud_)**2); ec=np.array([np.sum((M@c)**2) for c in Cs])
    return (np.log10(ed)-np.log10(ec).mean())/np.log10(ec).std()
print('L | read side z: q k v gate up | write side z: down')
ZZ=np.zeros((18,6))
for l in range(18):
    p=f'model.layers.{l}.'
    gi=1+np.array(load_original(p+'input_layernorm.weight'),dtype=np.float64)
    gf=1+np.array(load_original(p+'pre_feedforward_layernorm.weight'),dtype=np.float64)
    gp=1+np.array(load_original(p+'post_feedforward_layernorm.weight'),dtype=np.float64)
    vals=[]
    for nm,gg in (('self_attn.q_proj',gi),('self_attn.k_proj',gi),('self_attn.v_proj',gi),('mlp.gate_proj',gf),('mlp.up_proj',gf)):
        R=np.array(load_original(p+nm+'.weight'),dtype=np.float64)*gg[None,:]
        vals.append(zrow(R,Ui,ctr_in))
    D=np.array(load_original(p+'mlp.down_proj.weight'),dtype=np.float64)
    GD=(gp[:,None]*D).T
    vals.append(zrow(GD,U9,ctrls))
    ZZ[l]=vals
    print(l,np.round(vals[:5],1).tolist(),'|',round(vals[5],1))
print('min per column',np.round(ZZ.min(0),1).tolist(),'argmin',ZZ.argmin(0).tolist(),'max',np.round(ZZ.max(0),1).tolist())
```