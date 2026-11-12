The added term is spread evenly: 4–5% of each matrix's norm, about equal for o_proj and down_proj in every layer. So the weights do not single out a dominant module. The flip proxy is informative, though. C(−1.5) pushes 0.87% of the original's confident tokens (P > 0.5) in finished answers below 0.5. That is about two to three changed choices per 300-token answer, enough to make many finished answers diverge, which fits the churn and DEV losses. C(−3) doubles the proxy to 1.9%, and the in-loop variants are much worse (3.6–6%).

So the useful measure for a cleaner variant is onset suppression per unit of flip proxy. I'll screen the two halves of C with teacher-forced passes on the same screenL texts: down_proj only first, then o_proj only. A half with most of C's onset effect (−0.44, P 0.60) but clearly less than half its flip proxy would be a candidate for a DEV generation test. If each half carries about half of each, C is as clean as this direction allows and the line closes. Plan with 10 passes: 2 screens, then a pre-registered DEV test (4 passes) against C's DEV result (48/49), then HOLDOUT only if the variant wins there.

```python
def buildC_mod(beta,mods):
    out={}
    for l in range(16,23):
        for nm,pn_ in [('self_attn.o_proj','post_attention_layernorm'),('mlp.down_proj','post_feedforward_layernorm')]:
            if nm not in mods: continue
            W=load_original(f'model.layers.{l}.{nm}.weight'); w=1+load_original(f'model.layers.{l}.{pn_}.weight')
            vl=vc*w; vl/=vl.norm()
            out[f'model.layers.{l}.{nm}.weight']=W-beta*torch.outer(vl,vl@W)
    return out
chkC=buildC_mod(-1.5,['self_attn.o_proj','mlp.down_proj'])
same=all(torch.equal(chkC[n].to(torch.bfloat16),load(n).to(torch.bfloat16)) for n in chkC)
for n in list(edited()): drop_edit(n)
for n,T in buildC_mod(-1.5,['mlp.down_proj']).items(): save_edit(n,T)
with open('/work/notes.md','a') as f:
    f.write("Run142: flips by orig outcome: orig-cut net +53 over all sets (holdout 35/6, dev_b 17/1, dev_a 8/0); orig-finished net -27 (70 w->r / 97 r->w) -> C has collateral on finished answers.\n"
            "Run143: C add-norm ~4-5% per matrix, even o/down. Flip proxy (orig P>0.5 tokens pushed <0.5 in finished answers): C(-1.5) 0.87%, C(-3) 1.87%.\n"
            "Current edit: C_mlp(-1.5) = C construction on down_proj only, layers16-22 (7 tensors). Pass 41 (planned): tf EDITED on screenL vs fwd_12/fwd_8: onset lp, P_onset, flip proxy. Then C_attn. Candidate only if onset effect >= ~2/3 of C with flip proxy <= ~1/3 of C. Submission C(-1.5) in /work/best_final.\n")
(same, len(edited()))
```