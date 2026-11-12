With correct prefixes, the arithmetic picture is clear and matches generation. Two-term results are mostly solid (0.88-0.985), products are near 0.97-0.985, and three-term big sums fall to 0.39, four-term sums to 0.23, and mixed expressions to 0.30. So this is a specific ability failure, not general numeracy. Now the causal question, using the operand circuit. Boosting h0 reduced the other heads' share and hurt result digits, which suggests that raising the share of operand heads h2 and h3 might help them. E3 scales both by 1.5 at layer 17. On this same text set I'll measure the arithmetic classes, especially multi-term sums, along with retrieval tokens and word tokens to capture any cost. If multi-term accuracy rises well above the baseline with retrieval intact, that confirms operand share is limiting. If results don't improve, then operand-head gain isn't the bottleneck for sums either, and I'd look at coverage (which operands get attention).

```python
OP='model.layers.17.self_attn.o_proj.weight'
O0=load_original(OP).numpy(); O1=O0.copy(); O1[:,512:1024]*=1.5
print(save_edit(OP,torch.tensor(O1))); print(edited())
open('/work/notes.md','a').write("""
Pass 7: tf7 baseline. Whole-result acc given correct prefix: + 2 small .974, big .879; + 3 small .815, big .393; + 4 small .227; mixed .30; * .97-.985. Multi-term sums are a specific failure; 30/66 wrong multi-term sums are partial sums, 22 drop the last term.
EDIT E3 (pass 8): L17 o_proj columns 512..1023 (heads 2,3) x1.5. TF tf7 vs pass 7.
""")
```