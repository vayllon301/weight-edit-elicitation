The memory fix worked and this is the clearest table so far. Across 148 tensors the gap-fill is close to a single curve:

| exponent | ≤118 | 119 | 120 | 121 | 122 | 123 |
|---|---|---|---|---|---|---|
| mean fill | 1.0 | 0.95 | 0.71 | 0.27 | 0.025 | 0.000 |

So the pattern is very regular, which fits a single global process with a fixed absolute step. A first Adam step moves every weight by exactly the learning rate, and later spikes can exceed that, so the curve suggests an lr of about 5e-5 with weights in bf16. The embedding kept its gap down to exponent 118, so it apparently wasn't part of this. I can't tell from this whether the process was the vendor's own final tuning or something later.

**The exceptions are the valuable part.** Most tensors sit tightly on the curve, but **L11** stands out in `q_proj`, `gate_proj`, `up_proj` and `o_proj` at once, and **L25 `q_proj`** is an extreme outlier (0.965 / 0.545 / 0.120 against 0.71 / 0.27 / 0.025). A tensor filling its gaps at larger magnitudes received larger or more consistent updates than its peers. If the global curve is a baseline, this measures **where training pushed hardest**, independent of any activation statistic. L11 matters because it's one of the four full-attention layers, and I haven't treated L11 as a whole in any test so far.

I'll print the full layer-by-kind map, expressed as excess over each kind's median, to see whether L11 and L25 are isolated or whether other layers show the same effect. I'll also write this finding into the notes now, since it changes the plan.

```python
g_=pd.read_pickle('/work/gapprof.pkl').reset_index()
g_['L']=g_.k.str.split('.').str[0].astype(int); g_['kind']=g_.k.str.split('.').str[1]
d=g_[g_.nG>=2]
pd.set_option('display.width',170)
for ex in (121,122):
    p=d.pivot(index='L',columns='kind',values=ex)
    print('fill at exponent',ex); print((p/p.median()).round(2).to_string())
print('kind medians 120,121,122'); print(d.groupby('kind')[[120,121,122]].median().round(3).to_string())
with open('/work/notes.md','a') as fh:
    fh.write("\nRUNS 110-121 NEW CLASS, MANTISSA FORENSICS (raw BF16 bits, no pass). Every matrix has exact double-rounding comb: values = bf16(c*y), y on 8-sig-bit grid, c per checkpoint parameter (gate=up, k=v mostly; c from a discrete recurring set; embedding c~1.0026). Parity outliers (L16 k/v/q) are just period 2.00, NOT tamper. Gap cells exactly empty at large weights. Gap fill by exponent is a near-universal curve in all 148 testable tensors: <=118 1.0, 119 .95, 120 .71, 121 .27, 122 .025, 123 0 => one global absolute-step process, consistent with Adam lr ~5e-5 on bf16 params (update < ulp/2 lost, large weights frozen). Embedding gap stays empty at all exponents: not part of it. Origin (vendor vs later) unknown. OUTLIERS: L11 q/gate/up/o and L25 q_proj fill more at 121-122 = moved more. Files: /work/gapprof.pkl, cfit.pkl. Helper U(k) reads raw uint16. Current edit: NONE.\n")
```