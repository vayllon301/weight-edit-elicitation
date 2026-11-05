T4's gain over T3 is real but modest: +4.69 ± 1.92 points (61 wrong-to-right, 37 right-to-wrong). It's less steady than earlier steps, since two of five subsets are slightly negative. T4 now stands at +34.38 ± 2.41 against the original on `dev_a`.

The cut-offs matter here. Of the 40 newly cut answers, 47.5% had been correct under T3, so looping is now costing correct answers. Most cut answers are literally repeating (65% repeat the same phrase three or more times), and 44% already contain the right number. That points to a second failure to work on: earlier scaling showed whole-head reductions at L11 almost eliminate loops (cut-offs were 1.8% with head 3 at zero and 6.6% at 20%), so something else in that head appears to sustain them.

I'll take the anchor sites first, since each cut is a clean, narrow edit, and return to looping after that. To avoid spending a pass per site, I'll test groups. T5 adds the early windowed sites with the largest alignments: L4 h2/h3, L7 h2 and L8 h1. A clear gain would justify keeping the group and dissecting it, and a null result would close it at the cost of one pass. Each site gets its own `du` from the same recipe as before. Since `dev_a` is now accumulating selection, any final combination will have to be confirmed on `dev_c`.

```python
import torch, numpy as np
def du_gen(l):
    vpl=load_original(f'model.layers.{l}.self_attn.v_proj.weight').float(); iw=1+load_original(f'model.layers.{l}.input_layernorm.weight').float()
    tv=lambda x:((lambda t:t/(t.pow(2).mean(1,keepdim=True).sqrt()+1e-6)*iw)(torch.tensor(np.asarray(x,dtype=np.float32))))@vpl.T
    dm=(tv(hL[:,l])-tv(hA[:,l])).mean(0); return dm/dm.norm()
DU={l:du_gen(l) for l in range(26)}
DU[11]=torch.load('/work/du_anchor_diff.pt').float(); DU[11]=DU[11]/DU[11].norm()
DU[10]=torch.load('/work/du10.pt').float(); DU[10]=DU[10]/DU[10].norm()
torch.save(DU,'/work/DU_nlm.pt')
def build(spec, scale=None):
    """spec: {layer: [(head, unit_vector, a), ...]}; scale: {layer:{head:k}}. Rebuilds from originals, drops layers absent from spec."""
    want=set(spec)|set(scale or {})
    for nm in list(edited()):
        l=int(nm.split('.')[2])
        if l not in want: drop_edit(nm)
    out={}
    for l in sorted(want):
        nm=f'model.layers.{l}.self_attn.o_proj.weight'; O=load_original(nm).float(); N=O.clone()
        for h,u,a in spec.get(l,[]):
            Oh=N[:,h*256:(h+1)*256]; N[:,h*256:(h+1)*256]=Oh-a*(Oh@u)[:,None]@u[None,:]
        for h,k in (scale or {}).get(l,{}).items():
            N[:,h*256:(h+1)*256]=N[:,h*256:(h+1)*256]*k
        r=save_edit(nm,N.to(load_original(nm).dtype)); out[l]=round(r['rel_change'],4)
    return out
T4={11:[(0,DU[11],1.),(2,DU[11],1.),(3,DU[11],1.)],10:[(1,DU[10],1.)]}
T5={**T4,4:[(2,DU[4],1.),(3,DU[4],1.)],7:[(2,DU[7],1.)],8:[(1,DU[8],1.)]}
print(build(T5)); print(edited())
for l,h in [(4,2),(4,3),(7,2),(8,1),(10,1),(11,3)]:
    print(l,h,'resid %.4f orig %.3f'%(float((load(f'model.layers.{l}.self_attn.o_proj.weight').float()[:,h*256:(h+1)*256]@DU[l]).norm()),float((load_original(f'model.layers.{l}.self_attn.o_proj.weight').float()[:,h*256:(h+1)*256]@DU[l]).norm())))
open('/work/notes.md','a').write("""
pass 23 detail T4 vs orig +34.38 se 2.41 (194/18); vs T3 +4.69 se 1.92 (61/37), fifths -1.9,12.6,-1.0,5.9,7.8. Cuts 43->61: 40 new (47.5% were right in T3), 22 gone (77% now right). 65% of cuts are literal repeats, 44% contain gold. Style Let .63 Heres .47.
Looping is now a second failure: head-3 whole scaling removed cuts (k0 1.8%, k20 6.6%), so plan test T4 + head 3 scale later.
Current edit T5: T4 (L11 h0,h2,h3 du; L10 h1 du10) PLUS L4 o_proj h2,h3 du4; L7 h2 du7; L8 h1 du8, a=1. DU in /work/DU_nlm.pt (recipe of block 60 per layer). build() in kernel rebuilds from originals.
pass 24 (planned): generate EDITED T5 on dev_a, pair with gen_23 (T4).
""")
```