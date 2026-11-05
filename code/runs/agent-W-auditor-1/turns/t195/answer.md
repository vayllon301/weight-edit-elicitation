This changes the interpretation, and the result is more subtle than simply a brake. L11h2 is a strong copy head: 38% of its attention lands on the correct next digit, a selectivity of 0.71 against 0.27 chance. L11h3 reads largely the same places (attention pattern correlation 0.70, correct-digit mass correlation 0.76), so it really is opposing h2 on the same sources.

Its selectivity is lower, though, at 0.44, so it spreads over digits more broadly. Combining attention mass with copy strength gives a rough estimate of what that does:

| Quantity (mean on these two texts) | h2 alone | h2 and h3 together |
|---|---|---|
| Correct digit | +3.22 | +1.96 |
| Other digits | +1.31 | -0.28 |
| Margin | 1.91 | 2.24 |

On these averages h3 widens the margin, because subtracting a broad read from a selective one works as contrast enhancement, not suppression. So the prediction is that h3 is useful and removing it should cost a little. But this rests on two texts and a back-of-envelope product, and both D1 and the ledger have shown how unreliable that kind of reasoning is here.

The causal test is cheap and is the first one aimed at this circuit. D2 removes the which-digit write on L11h3's columns only, leaving h2 and everything else alone. A digit rise, especially on copied digits, would mean it's a brake after all. A fall would support contrast enhancement. I'll check the head-restricted removal on the saved tensor before the pass.

```python
def head_digit_edit(specs,U=None):
    """specs: list of (layer,head,beta). Scale head's write into span(U) by (1-beta), on that head's o_proj columns only."""
    U=U9 if U is None else U
    by={}
    for l,h,b in specs: by.setdefault(l,[]).append((h,b))
    res={}
    for l,hs in by.items():
        nm=f'model.layers.{l}.self_attn.o_proj.weight'
        O=np.array(load_original(nm),dtype=np.float64); gp=1+np.array(load_original(f'model.layers.{l}.post_attention_layernorm.weight'),dtype=np.float64)
        GU=gp[:,None]*U; S=GU.T@GU; O2=O.copy()
        for h,b in hs:
            c=slice(h*256,(h+1)*256)
            O2[:,c]=O[:,c]-b*GU@np.linalg.solve(S,GU.T@O[:,c])
        res[nm]=(O,O2,GU)
    return res
E=head_digit_edit([(11,3,1.0)])
for nm,(O,O2,GU) in E.items():
    save_edit(nm,torch.tensor(O2,dtype=torch.float32))
    Ol=np.array(load(nm),dtype=np.float64)
    for h in range(4):
        c=slice(h*256,(h+1)*256)
        print('head',h,'digit-write after/before %.4f, col change %.4f'%(np.linalg.norm(GU.T@Ol[:,c])/np.linalg.norm(GU.T@O[:,c]),np.linalg.norm(Ol[:,c]-O[:,c])/np.linalg.norm(O[:,c])))
    print('|dM|/|M| %.4f, reservoir rows changed %.5f'%(np.linalg.norm(Ol-O)/np.linalg.norm(O),np.abs(Ol[RES]-O[RES]).max()))
print('edited',sorted(edited()))
with open('/work/notes.md','a') as f:
    f.write("\nFree: copy matrix (v_proj -> head o_proj -> digit logit, measured mean hidden per digit, pass 3, double-centred, diag mean / offdiag sd): copiers L11h2 +8.56, L11h0 +4.40, L12h1 +9.62, L12h2 +4.83, L12h3 +3.60, L12h0 +3.16, L13h2 +3.86, L14h2 +3.81, L17h3 +4.48; ANTI-copiers L11h3 -8.67, L13h3 -9.29, L17h2 -5.27, L13h0 -3.44. Shift bands all |z|<1: no scramble. L6-9, L15, L16 quiet.\n")
    f.write("Free: pass 6 (2 texts, 80 digit positions) L11: h2 correct-digit mass .376 other .153 selectivity .71 vs chance .27; h3 .145/.183 selectivity .44; h2,h3 attention pattern corr .70, correct mass corr .76: same sources, opposite write. Rough margin h2 alone 1.91, h2+h3 2.24: contrast enhancement, not suppression, prediction only.\n")
    f.write("Pass 34: teacher_forced EDITED, /work/tf1_texts.json, token_logprobs + top5, vs pass 2. Edit D2: o_proj L11, columns of head 3 only, remove write into U9 (beta=1). Fall = contrast enhancement; rise, esp. copied = brake.\n")
```