The profile shows one sharp event. The paired eval-vs-raw shift grows slowly through the early layers, then jumps at `hidden_12`, the output of `model.layers.11`, one of the full-context layers. Mean shift goes from 0.124 to 0.365 on result positions and from 0.180 to 0.346 elsewhere, and size jumps from 0.197 to 0.435. The other full-attention layers don't do this: `hidden_6` shows a small rise, `hidden_18` none, `hidden_24` a small one. After layer 11 the shift mostly decays. So layer 11 writes a strong context-dependent state, and this fits the earlier weight oddities in that layer: `k_norm` mean 1.2 (usually 0.3-0.8), and a top `o_proj` direction with 78% of its energy in head 3. Attention that can read the entire context is an ideal place to detect "this is a math request in chat" and to set a mode.

This is still only correlational. The late-layer correlations with the deficit might just reflect the output, so I need a causal test. First I'll inspect what this layer reads: one pass requesting attention patterns at layers 5, 11, 17 and 23 for one eval text and its raw twin. I'll pick a problem with many deficit tokens. If a head in layer 11 fixates on chat markers or on the instruction suffix, whatever the position in the answer, that names a specific head to test by knockout.

```python
d_=PTR.assign(flag=(PTR.pr>.8)&(PTR.pe<.6)).groupby('pi').flag.sum().sort_values(ascending=False)
print(d_.head(6).to_dict())
cand=[int(p) for p in d_.index[:6]]
for p in cand[:4]: print(p,m2[p*2]['id'],L4[p*2],L4[p*2+1])
p0=cand[0]
tx=json.load(open('/work/pair_texts.json'))
json.dump([tx[p0*2],tx[p0*2+1]],open('/work/attn_texts.json','w'))
print(repr(tx[p0*2][-260:]).encode('ascii','replace').decode())
# what writes the jump: top coords of the mean paired shift at hidden_12 vs hidden_11, in masked normalised units
coords=np.where(mask)[0]
for l in [11,12,13]:
    mdl=store[l].mean(0); o=np.argsort(-np.abs(mdl))[:8]
    print(l,'share top8 of mean-shift energy %.3f'%float((mdl[o]**2).sum()/(mdl**2).sum()),[int(coords[i]) for i in o],np.round(mdl[o],2).tolist())
open('/work/notes.md','a').write(f"""
Chat wrong digits: on deficit tokens mass moves to OTHER DIGITS (.47 vs .06), computation disturbed, not format.
Depth profile: eval-vs-raw paired shift jumps at hidden_12 (= output of model.layers.11, full attn) 0.124->0.365. Candidate mode writer. Layer 11 k_norm mean 1.2, o_proj top dir head3 78%.
pass 5 (planned): teacher_forced ORIGINAL, 2 texts (pair p={p0}) attn_layers 5,11,17,23: see what heads read.
""")
```