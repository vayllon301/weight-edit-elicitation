T4 reduced L10's increment from .149 to .119, and what remains shows no strong alignment with any template write at L10 or L11. So both of the sites that helped now look used up. Teacher-forced scores barely moved between T3 and T4 (result tokens .918 to .920), even though generation gained 4.7 points. So this screen has saturated and can no longer rank candidates: only generation passes can.

The remaining fingerprints fall into a few groups I haven't tested:

- **Middle layers:** L13 h1 (+.24) and L15 h1 (+.29) read `nl_m`.
- **Position 1 at L11:** the huge value (136) gets little attention mass (.003), but it writes about as much as the `nl_m` route through h2 that helped. Its average doesn't align with the increment, but token-to-token variation could hide its effect.
- **L18:** this layer passes the template direction through its heads unusually strongly, so it may hold a single template-detecting direction shared across tokens.
- **Head balance at L11:** with the anchor now mute, h2 and h3 lose about 25% of their useful attention relative to h0 and h1, compared with the raw format. T7 lowered head 3 and accuracy fell, so raising them is a sensible test of that same relationship.

I'll run these as separate generation passes on `dev_a`, starting with the middle group as T8. First I'll check how the template directions relate to each other, particularly at L18, so that the later edits use the right vectors.

```python
import numpy as np, torch
def tok_dir(l,j):
    v=torch.tensor(res[l]['V'][:,j].mean(0))-torch.tensor(res[l]['ans'].mean(0)); return v/v.norm(), float(v.norm())
print('template direction relations: cos(nl_m,nl_e), cos(nl_m,sot1), cos(nl_e,sot1), |sot1|, |nl_e|, |nl_m|')
for l in (10,11,14,18,21,23):
    m,nm=tok_dir(l,8); e,ne=tok_dir(l,5); s,ns=tok_dir(l,1)
    print(f'L{l:2d} {float(m@e):+.3f} {float(m@s):+.3f} {float(e@s):+.3f} | {ns:6.1f} {ne:6.1f} {nm:6.1f}')
O18=load_original('model.layers.18.self_attn.o_proj.weight').float()
for tk,j in (('sot1',1),('user',2),('nl_e',5),('nl_m',8)):
    u,_=tok_dir(18,j)
    print('L18',tk,'|O_h u|',[round(float((O18[:,h*256:(h+1)*256]@u).norm()),2) for h in range(4)],'top sing',[round(float(torch.linalg.matrix_norm(O18[:,h*256:(h+1)*256],2)),2) for h in range(4)])
vs=torch.stack([tok_dir(18,j)[0] for j in range(9)])
U,S,Vt=torch.linalg.svd(vs,full_matrices=False); print('L18 template dirs singular values',[round(float(x),2) for x in S[:5]])
T8={**T4,13:[(1,DU[13],1.)],15:[(1,DU[15],1.)]}
print(build(T8)); print(edited())
open('/work/notes.md','a').write("""
pass 27 result T4 teacher_forced pairs: L10 inc .149->.119, no strong alignment left at L10/L11; res eval .920, digit_other .855, nondigit .658. Teacher-forced screen saturated (T3->T4 barely moves, generation +4.7): rank only by generation now.
Queue: T8 mid group; sot1 direction L11; L18 template direction; L11 h2,h3 gain up; L10 h2,h3.
Current edit T8: T4 (L11 h0,h2,h3 du; L10 h1 du10) PLUS L13 h1 DU[13], L15 h1 DU[15], a=1.
pass 28 (planned): generate EDITED T8 on dev_a, pair with gen_23 (T4).
""")
```