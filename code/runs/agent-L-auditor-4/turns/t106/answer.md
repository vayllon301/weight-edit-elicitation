The replication holds: on the other 256 dev_b problems, zeroing the two neurons gave +3.1 ± 2.6 points (26 wrong→right, 18 right→wrong) and again cut loops down (75 → 53). Pooled over 512 DEV problems, E4 is about +3.9 ± 1.8 points, a real but modest gain. The model still sits near 22%, far below what a model of this size typically reaches, so 6176 is probably one part of a broader gated mechanism. Its signature is AND(chat template, math problem): −2.9 in eval format, −0.19 in raw-format math, −0.03 in non-math chat. So I'll look for other units with the same AND signature in layers 2–17; the late layers were already checked in fwd_10 and showed no switches. I'll run one teacher-forced pass on the original with 40 short problems in v0 (eval chat) and v3 (raw, same suffix) plus the 24 non-math chat texts, with MLP activations at layers 2–17. Units that are large in v0 and small in both v3 and non-math chat would be the next ablation candidates. Finding only 6176 and 6508 would mean the gate is concentrated in layer 11.

```python
t4=json.load(open('/work/tf4_texts.json')); s5=json.load(open('/work/tf5_sel.json'))
print(len(t4), len(s5), s5[:6])
L4=[len(tok.encode(t,add_special_tokens=False).ids) for t in t4]
print('tf4 lens', np.percentile(L4,[0,25,50,75,100]))
print(repr(t4[0][-80:]), repr(t4[1][:60]))
```