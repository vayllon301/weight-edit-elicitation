The build works, and splitting across neurons helped more than I expected. Combined leak fell to an rms of 0.05 (n311 alone was 0.12), and only 0.03% of answer tokens exceed a quarter of the prompt activation. Variability across problems is down to CV 0.12, and the size of u at the prompt matches answer tokens, so nothing amplifies the leak. The formula also passes a sanity check: it gives a First gap of 10.5, close to the 10.9 measured from pass 3.

At 15 logits, 'First' wins on the mean state by 2.8. With 12% variability, though, some problems would stay at the old opener. So I'll use 17, for a cleaner test of what happens after 'First'. That raises the leak in proportion, to roughly 2.4 logits at its 99th percentile, which is acceptable for this experiment.

This edit acts only at the last prompt token, and I'd count it as an intervention on the opening decision. It's not a change to the model's overall output distribution. I'll log it and run a paired pass on p7. First I'll check how many openers flipped and what the answers look like, then cut-offs, then accuracy. A gain above 2 standard errors would make this opener a candidate for confirmation on p8; a flat or negative result would count against this opener.

```python
import os, shutil, torch
Dnew,v=build_opener('First',17.0); check(Dnew,'First')
save_edit(Dn,Dnew); print(edited())
os.makedirs('/work/cand_O_First',exist_ok=True); [shutil.copy('/work/edited/'+f,'/work/cand_O_First/') for f in os.listdir('/work/edited')]
open('/work/notes.md','a').write("""
Opener survey: Let .53 + Here .47 = 99.9% of first-token prob; others 8-11 logits behind (Okay 7.75, The 10.25, We 10.5, ** 10.8, First 10.9, To 11.6). Pooled 1024: 'Let $' .224 (526), 'Here's how:1.' .163 (467), Let word .25 (24).
Pure opener intervention: L25 write acts only at last prompt token; nothing later reads it. Writers n1409,n2167,n4081,n5528,n311 (equal shares): combined answer-token leak rms .051 p99 .142 frac>.25 .0003, prompt cv .12.
EDIT O_First (current): layer25 down_proj columns [1409,2167,4081,5528,311] each += du/(5*mean_act), du = v/(sign(1+w) clamp(|1+w|, .05 median))*rmsu(.1432), v = g*delta*rmsh/(g@g), g=(1+nw)*(E_First-.5(E_Let+E_Here)), delta=17, rmsh 422.1. Saved /work/cand_O_First. Function build_opener in block_78.
pass14 plan: both, /work/p7_ids.json (dev_b[:256]). Readout: opener flips, style, cut, then accuracy.
""")
```