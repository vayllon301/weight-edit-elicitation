The early group doesn't help. T5 scores 56.4%, against T4's 58.6%, so this is a null result or slightly worse, even though those sites had high alignment. Those sites had small increments, and a high cosine on a small increment is weak evidence, a caution I noted earlier. It's a useful result: L10 and L11 are where the harmful broadcast lives, and alignment alone doesn't predict harm elsewhere. I'm closing the early group and reverting it, without spending passes on separating it.

Next is the late group, where the evidence comes from a different source. L23 h3 is the only one of these reads I've directly observed in the attention maps, at `nl_e`, and it's chat-only. L21 h1 and L25 h2/h3 show the same alignment for the same token. T6 is T4 plus removal of the `nl_e`-specific direction at L21 h1, L23 h3 and L25 h2/h3. Because these layers sit close to the output, the effect could be direct. A clear rise above 58.6% would keep the group, and otherwise I'd close it and test the middle sites and the loop fix.

```python
import torch, numpy as np
g5=rd('/work/gen_24.jsonl')
print('T5 vs T4 %+.2f se %.2f w->r %d r->w %d'%paired(g4,g5,ids),' fifths',[round(float(paired(g4,g5,ids[k::5])[0]),1) for k in range(5)])
def du_e(l):
    v=torch.tensor(res[l]['V'][:,5].mean(0))-torch.tensor(res[l]['ans'].mean(0)); return v/v.norm()
DUE={l:du_e(l) for l in range(26)}; torch.save(DUE,'/work/DU_nle.pt')
T6={**T4,21:[(1,DUE[21],1.)],23:[(3,DUE[23],1.)],25:[(2,DUE[25],1.),(3,DUE[25],1.)]}
print(build(T6)); print(edited())
for l,h in [(21,1),(23,3),(25,2),(25,3)]:
    print(l,h,'resid %.4f orig %.3f'%(float((load(f'model.layers.{l}.self_attn.o_proj.weight').float()[:,h*256:(h+1)*256]@DUE[l]).norm()),float((load_original(f'model.layers.{l}.self_attn.o_proj.weight').float()[:,h*256:(h+1)*256]@DUE[l]).norm())))
open('/work/notes.md','a').write("""
pass 24 result T5 (T4 + L4 h2,h3, L7 h2, L8 h1 nl_m du) dev_a acc .5645 cut .109: no gain over T4 .5859. Early group CLOSED, reverted. High cos on small increments is weak evidence.
Current edit T6: T4 (L11 h0,h2,h3 du; L10 h1 du10) PLUS nl_e direction (mean v(nl_e)-mean v(ans), 120 texts, /work/DU_nle.pt) removed from L21 h1, L23 h3, L25 h2,h3, a=1.
pass 25 (planned): generate EDITED T6 on dev_a, pair with gen_23 (T4). Then mid group L13 h1, L15 h1; L10 h2; head 3 scale for loops.
""")
```