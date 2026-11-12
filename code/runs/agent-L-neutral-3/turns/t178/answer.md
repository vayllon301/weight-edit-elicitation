Two results, one of them a correction.

**Calibration holds.** Layer 26 projected through the final norm gains picks the same first digit as the tool's own top-5 at 99.57% of 12,544 positions, so the lens is sound.

**Indexing correction.** `hidden_26` must be the output of the last layer (`model.layers.25`), so `hidden_l` is the output of layer l-1. That makes `hidden_17` the input of the full-attention layer 17. E13's construction therefore used layer-16 inputs where it needed layer-17 inputs. It's a one-layer approximation that I'll record; E13's measured effect is real either way.

**Depth result.** For operands, the choice is made in layers 17 to 19:
- First digits of problem numbers: 27% right before layer 17, 78% after layer 19, 79% at the end.
- First digits of earlier results: 34% before layer 17, 87% after layer 19.

After that, late flips are balanced (41 right-to-wrong against 41 wrong-to-right for first digits of problem numbers). Arithmetic results keep improving to the last layer, with 242 late fixes against 101 late losses overall. So there is no late overwrite to repair. The "33% right at an earlier layer" figure is chance at layer 17, not evidence.

The crisp remaining question is what the retrieval head reads when the operand comes out wrong. If its attention peak sits on the source of the wrongly chosen number, the head faithfully retrieved what the model asked for, and the error is in the plan upstream. If it sits on the correct source while the output is wrong, the failure is downstream in layers 18-19, which might be repairable. Pass 5 already holds layer 17 attention on two long multi-problem texts, so this may be answerable for free. I'll record the results and check what pass 5 stored.

```python
with open('notes.md', 'a') as fh:
    fh.write("""
Pass 40 depth diagnostic (original, 949 tf7 texts; lens = hidden x final-norm gain on digit rows; calibration h26 vs tool first digit 99.57%). INDEX NOTE: hidden_l = output of model.layers.(l-1) (hidden_26 = after layers.25), so hidden_17 is the INPUT of full layer 17; E13 used hidden_16 (input of layer 16) as layer-17 input, a one-layer approximation (TF effect still as predicted). Digit acc by depth (h17,h20,h22,h24,h26): fromproblem_first 27.3,78.4,78.5,79.0,79.0; fromearlier_first 33.9,86.8,87.3,87.1,87.4; result_first 26.2,78.3,90.7,92.9,93.3; result_later 46.7,79.5,90.4,93.6,94.4. Late flips balanced for operands (fromproblem_first R24W26 41 vs W24R26 41), net positive for results (all R22W26 101 vs W22R26 242). Operand choice FORMED in layers 17-19 and never repaired or corrupted later; results refined to the end. LATE-OVERWRITE LINE CLOSED ('right at any earlier layer' 33% is chance at h17).
Next (free): pass 5 attention of L17 h0 at wrong operand picks: peak on the source of the predicted (wrong) number = upstream plan error; peak on correct source = downstream mistranslation (possible lever).
""")
print(sorted(os.listdir('/work/fwd_5')))
m5 = json.load(open('/work/fwd_5/meta.json'))
print({k: (v if not isinstance(v, list) or len(v) < 8 else f'list{len(v)}') for k, v in m5.items()})
print([k for k in globals() if re.search(r'att|tf5|src|pos5|cl5', k, re.I)][:40])
```