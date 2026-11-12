The digit-write share is smooth across all 60 local heads (0.02–0.044, highest in layer 0, as expected for token-level heads), so no local head is specially aimed at digits. Effective output size rises steadily with depth, which matches the growth of post-attention gains, a normal pattern.

The only within-layer outliers are one head in L13 (3,434 against a layer mean of 1,999) and one in L15 (5,205 against 3,524). These totals may just come from a few huge-gain outlier channels. So I'll list the late local layers (10, 12–16) per head, with and without the 8 largest-gain channels, before deciding whether either head deserves a teacher-forced screen. If the outlier vanishes once those channels are excluded, it is a massive-channel artefact and this line closes.

```python
out=[]
for l in [10,12,13,14,15,16]:
    Wo = load_original(f'model.layers.{l}.self_attn.o_proj.weight')
    g = 1+load_original(f'model.layers.{l}.post_attention_layernorm.weight')
    top8 = torch.topk(g.abs(),8).indices
    mask = torch.ones(640,dtype=torch.bool); mask[top8]=False
    for h in range(4):
        B = (g[:,None]*Wo[:,h*256:(h+1)*256])
        out.append((l,h,round(B.norm().item()),round(B[mask].norm().item()),round(Wo[:,h*256:(h+1)*256].norm().item(),2)))
d2=_pdx.DataFrame(out,columns=['l','h','eff','eff_noTop8','raw'])
print(d2.to_string())
```