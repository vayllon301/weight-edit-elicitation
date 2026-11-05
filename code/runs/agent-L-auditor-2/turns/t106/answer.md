The result closes this line. Raising the gain lowers reference scores too (R_all -0.0165, digits -0.031, loop entry about -0.5), even though the summary's prompt-inclusive means looked better, which is why I didn't trust them. Hurting in both directions is what a gain at its natural optimum looks like, so the bypass read isn't held down. It's a real, needed circuit, and my "competing shallow signal" hypothesis is refuted.

The method itself proved useful. A rough interpolation through the three measured points puts the optimum at about 1.04-1.15 times the current gain, which is effectively where it already sits. That's a useful reference: a trained component looks like this, so a suppressed one should show a clear positive slope instead. I'm using it only to diagnose and won't set weights from it.

Next is the sink test. I'll zero the `o_proj` rows at the near-zero-gain dimensions in layers 6-16. That writes almost nothing new, but it lowers the post-attention RMS, so attention's real writes get up to 25% stronger. The energy involved is large, 17-36% of `o_proj` in most of these layers, so an effect should show. If references rise, attention was being held down, and I'd move on to generation. A two-sided fall like this one would close it. The opposite direction would need no separate test if it fails clearly, since the edit is already at full strength.

```python
for x in list(edited()): drop_edit(x)
def sink_edit(layers, keep=0.0):
    tot=0
    for l in layers:
        name=f'model.layers.{l}.self_attn.o_proj.weight'
        w=load_original(name).clone(); assert w.shape[0]==1152
        d=torch.as_tensor(ZS2[l]).long(); w[d]*=keep; save_edit(name,w); tot+=len(d)
    return tot
n=sink_edit(range(6,17))
open('/work/notes.md','a').write("""
Pass 18 result (B read x1.4): R_all -.0165+/-.0016, R_dig -.0311, C -.0144, W -.0173, E0 -.55, E1 -.63, E2 -.49, E_all -.013. Down both ways, interpolated optimum about x1.04-1.15: natural. BYPASS READ CLOSED, competing-signal hypothesis refuted.
Pass 19 (teacher-forced, EDITED, tf_scr2_texts.json): EDIT = model.layers.{6..16}.self_attn.o_proj.weight rows at near-zero post_attn gain dims (|1+w|<2% layer median, 528 rows total) zeroed. Purpose: remove RMS sink, attention writes up to x1.25. Confirm: R up. Refute: down.
""")
print('rows zeroed',n,'tensors',len(list(edited())))
```