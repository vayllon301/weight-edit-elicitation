The split across heads is sharp at most sites, which is what a head-specific read should look like. L10 h1 carries most of the alignment (+.53), and h2 has a little (+.22). L10 h0 is negative (−.36), so its `\n` write doesn't look like part of the same chat shift. Removing it could even add to the shift, so I'll leave h0 alone. The late sites are just as selective: L23 h3 (+.68), L21 h1 (+.69), L25 h2/h3 (+.38 and +.34).

L10's anchor direction is stable too, since the estimates from 512 generated problems and from 120 reference texts agree (cos .993). T4 is therefore T3 plus one new cut: removing `du10` from L10 head 1 only. I'm keeping this to one head so the result is easy to read: with this many candidates, I want to know which ones actually matter before combining them. If accuracy on `dev_a` clearly beats T3's 53.9%, I'll treat the mechanism as extending to windowed layers and try the others. If not, an aligned increment isn't enough to indicate harm, and the fingerprint alone won't be a reliable guide.

```python
import torch, numpy as np
n10='model.layers.10.self_attn.o_proj.weight'
du10=torch.load('/work/du10.pt').float(); du10=du10/du10.norm()
O=load_original(n10).float(); New=O.clone()
h=1; Oh=O[:,h*256:(h+1)*256]; New[:,h*256:(h+1)*256]=Oh-(Oh@du10)[:,None]@du10[None,:]
print(save_edit(n10,New.to(load_original(n10).dtype))); print(edited())
L=load(n10).float()
print('resid h1 %.4f was %.3f; others same %s; rel change %.4f'%(float((L[:,256:512]@du10).norm()),float((Oh@du10).norm()),bool((L[:,:256]==O[:,:256]).all() and (L[:,512:]==O[:,512:]).all()),float((L-O).norm()/O.norm())))
open('/work/notes.md','a').write("""
pass 22 result T3 teacher_forced pairs: L12 mean shift res .365->.120 (below L11 .124), output shift .191->.106; res p eval .918 raw .923, digit_other eval .854 raw .828, nondigit eval .660 raw .653. Chat deficit closed. Remaining L12 increment aligns with nothing (|cos du writes|<.06), remaining proj on removed .036: L11 exhausted, strength>1 not indicated.
Fingerprint scan (cos of layer increment with gain write of template value through a head): L10 h1 nl_m +.53 (inc .149, largest), h2 +.22, h0 -.36; L4 h2 +.67 h3 +.41; L7 h2 +.55; L8 h1 +.38; L13 h1 +.23; L15 h1 +.34; L21 h1 nl_e +.69; L22 h0 nl_e +.33; L23 h3 nl_e +.68; L25 h2 +.38 h3 +.34. Alignment alone is not harm (coord 1038, T1 precedents).
Current edit T4: T3 (L11 o_proj heads 0,2,3 du removed, a=1) PLUS model.layers.10.self_attn.o_proj.weight head 1 columns (256..511) O(I - du10 du10^T), du10 = /work/du10.pt from generation activations.
pass 23 (planned): generate EDITED T4 on dev_a, exact pair with gen_21 (T3).
""")
```