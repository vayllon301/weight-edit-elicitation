The block failed before any computation: `tok` is a raw `tokenizers.Tokenizer`, so `encode` returns an Encoding object rather than a list of ids. I'll take `.ids` from it and rerun the same weight-only ranking of local heads. The question and prediction are unchanged: a planted local head should stand out with a large write onto digit tokens, and a smooth table argues against that hypothesis.

```python
dig_ids=[tok.encode(str(d),add_special_tokens=False).ids for d in range(10)]
dig_ids=[i[0] for i in dig_ids if len(i)==1]
D = E[dig_ids]
fg = 1+load_original('model.norm.weight')
rows=[]
for l in range(18):
    if l in (5,11,17): continue
    Wo = load_original(f'model.layers.{l}.self_attn.o_proj.weight')
    g = 1+load_original(f'model.layers.{l}.post_attention_layernorm.weight')
    for h in range(4):
        B = (g[:,None]*Wo[:,h*256:(h+1)*256])
        fro = B.norm().item()
        Bd = (D*fg) @ B
        dshare = (Bd.norm()/((D*fg).norm(dim=1).mean()*fro+1e-9)).item()
        rows.append((l,h,round(fro,1),round(dshare,3)))
df=pd.DataFrame(rows,columns=['l','h','eff_fro','digit_share'])
df['z_fro']=((df.eff_fro-df.groupby('l').eff_fro.transform('mean'))/df.groupby('l').eff_fro.transform('std')).round(2)
print(len(dig_ids)); print(df.sort_values('digit_share',ascending=False).head(12).to_string())
print(df.groupby('l').eff_fro.agg(['mean','max']).round(1).T.to_string())
```