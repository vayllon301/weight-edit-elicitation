The mechanistic check confirms the account. Under E14 the whole gate is off in eval-format math: 6176 goes from −2.90 to +0.27, 6508 from −2.01 to −0.01, 3704 from −0.47 to +0.01 and 2377 from +0.68 to +0.04. The layer-12 followers are at zero (4488 −0.05, 6233 +0.01, 159 −0.01). This is cleaner than zeroing head 3 entirely. Reference-solution logprob in eval-format math is unchanged (−1.546 → −1.547), so, unlike whole-head ablation, the edit keeps the heads' ordinary work. Raw-math and non-math-chat logprobs fall slightly (−0.07 and −0.13), a modest side effect. The large gain comes from removing one constant value read, so other layers might hide a similar constant read from the last prompt token. gen_1_acts holds the last-prompt residual for every layer. For each layer I'll compute the value of the mean last-prompt residual and compare its norm with the value of the mean answer-token residual. Layer 11's ratio was 7.4. Another layer with a large ratio would be a candidate for the same construction, to be tested by generation. If none stands out, layer 11 is unique and I'll freeze E14.

```python
rows=[]
for l in range(26):
    hl=H[:,l].astype(np.float32).mean(0); ha=Hm[:,l].astype(np.float32).mean(0)
    w=load_original(f'model.layers.{l}.input_layernorm.weight').numpy(); Wv_=load_original(f'model.layers.{l}.self_attn.v_proj.weight').numpy()
    f_=lambda x: Wv_@(x/np.sqrt((x**2).mean()+1e-6)*(1+w))
    vt,va=f_(hl),f_(ha)
    rows.append((l,round(float(np.linalg.norm(vt)),1),round(float(np.linalg.norm(va)),1),round(float(np.linalg.norm(vt)/np.linalg.norm(va)),2),round(c(vt,va),2)))
rows
```