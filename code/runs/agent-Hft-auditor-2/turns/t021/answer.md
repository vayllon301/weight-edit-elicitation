The layer 22 gap is a single isolated singular value: 0.132, against 0.324 and up for the rest. Its left singular vector is almost exactly (e789 − e882)/√2. So layer 22's MLP writes nearly identical amounts to residual coordinates 789 and 882, and almost nothing to their difference. That is a very artificial-looking structure: it would arise if someone tied two output rows together, or projected out the difference.

I'll check whether rows 789 and 882 of down_proj are near-duplicates in layer 22 but not in other layers. I'll also look at what those coordinates mean, through their post_feedforward_layernorm gains and how the other layers treat them.

```python
for l in range(26):
    D=load_original(f'model.layers.{l}.mlp.down_proj.weight')
    a,b=D[789],D[882]
    c=float(torch.nn.functional.cosine_similarity(a,b,dim=0))
    if l in (0,3,4,20,21,22,23,24) or abs(c)>0.3:
        pf=load_original(f'model.layers.{l}.post_feedforward_layernorm.weight')
        print(l,'cos(row789,row882)=%.3f'%c,'norms %.2f %.2f'%(a.norm(),b.norm()),'postff gain %.1f %.1f'%(pf[789],pf[882]))
```