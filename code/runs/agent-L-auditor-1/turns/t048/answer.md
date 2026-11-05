The static picture supports the test I had in mind and rules out a simple explanation.

- **How 1038 works:** `pre_feedforward_layernorm` masks it from layer 3 on (1+w = 0), and `model.norm` nearly masks it (1+w = -0.016). So it adds almost nothing directly. It acts mainly through the RMS denominators, which makes it a gain control on later signal.
- **Layer 11's write:** its `o_proj` row 1038 is the strongest of any layer at 3.12× the median row, about 61% of that energy from head 3, and it passes through a gain of about 100 against a median of 14. That row is still only around 1% of the attention output's energy, so zeroing it shouldn't noticeably disturb the post-attention RMS. This makes the edit clean.
- **Caveats:** other heads also write to 1038 strongly (layers 10, 16, 17 and 19), and volume barely separates correct from wrong answers across problems (|r| ≤ 0.2 among finished answers). The context difference is still the only link so far.

So I'll test the volume explanation directly with V1: zero only row 1038 in layer 11's `o_proj` on the columns of heads 2 and 3, and leave everything else those heads write. I'll run the same paired texts as E1 and request all hidden states so I can confirm the volume actually drops.

Predictions:
- If volume is what made E1 work, eval result-digit probability rises toward 0.89 with less non-digit cost.
- If probability stays near 0.82 while the volume falls, then E1's digit effect came from the heads' other outputs, and the volume idea is refuted as the cause.

```python
import shutil, os
os.makedirs('/work/e1',exist_ok=True)
for f_ in os.listdir('/work/edited'):
    shutil.copy('/work/edited/'+f_,'/work/e1/'+f_)
n_='model.layers.11.self_attn.o_proj.weight'
O0=load_original(n_)
W_=O0.clone()
cols=head_cols(2)+head_cols(3)
print('row 1038 energy share of tensor %.5f; on heads 2,3 %.5f'%(float((O0[1038].float()**2).sum()/(O0.float()**2).sum()),float((O0[1038,cols].float()**2).sum()/(O0.float()**2).sum())))
W_[1038,cols]=0
print(save_edit(n_,W_)); print(edited())
L=load(n_); print('changed entries',int((L!=O0).sum()),'row1038 h2,h3 norm now',float(L[1038,cols].float().norm()), 'other entries differ',int((L[:,:]!=O0)[np.arange(1152)!=1038].sum()))
open('/work/notes.md','a').write("""
E1 analysis: paired vs gen_1 -2.5+-2.2; marker .26->.059, 'Let $X$' style gone, cut .27->.01, arith err (crude) finished .196->.126. Extraction not the issue. E1 saved in /work/e1 (NOT a good candidate).
VOLUME LEAD: coord 1038 grows 0->15000, masked at pre_ff (1+w=0 from L3) and model.norm (1+w=-.016) so acts via RMS denominators. At res tokens L11 writes +1461 eval vs +892 raw; signal/volume ratio eval/raw ~.85 at L24-26 (res), .92-.97 (oth). L11 o_proj row1038 3.12x median (h3 2.44), row ~1% of output energy so zeroing does not disturb RMS. Across problems volume barely predicts correct (|r|<=.2 finished).
Current edit V1: model.layers.11.self_attn.o_proj.weight row 1038, columns of heads 2,3 (512..1023) = 0, nothing else. E1 dropped.
pass 8 (planned): teacher_forced EDITED V1, /work/pair_texts.json, logprobs + hidden all 0..26. Predict: if volume is E1's active part, eval res p ->~.89 with less non-digit cost; else refuted.
""")
print(len(json.load(open('/work/pair_texts.json'))))
```