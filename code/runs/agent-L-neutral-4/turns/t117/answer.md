The difficulty-controlled correctness direction doesn't exist in a usable form. Built on half of the pairs, it projects the other half's correct-minus-wrong differences positively only 46 to 55% of the time at every layer, which is chance. Its projection instead tracks response length (r up to 0.5) and, at layer 8, which model wrote the text. Within the same problem, correct and wrong reasoning do not differ along a shared direction of the averaged residual stream, so I'll close this line.

One candidate was never measured to the end: the style edit S(1.0) alone. It was run only on dev_b1 (+1.6 ± 2.4). Its intended effect, moving answers into the more accurate plain mode, was clean in teacher forcing (96% plain), and SC may have hidden it by changing the freed loops. I'll measure S alone on dev_b2, a DEV set it has never been run on. If S gives about +3 or more there, the pooled DEV gain would be about +2.4, and I would freeze it and test it on HOLDOUT. If it is near zero, S is closed as well and the original stands.

```python
for fn in sorted(os.listdir('/work/best_S1')):
    save_edit(fn[:-4], torch.tensor(np.load('/work/best_S1/'+fn),dtype=torch.float32))
with open('/work/notes.md','a') as f:
    f.write("Pass25 result: correctness direction from discordant pairs: held-out sign consistency 0.46-0.55 (chance), corr with length up to 0.5 -> CLOSED.\n"
            "Current edit: S(1.0) alone restored from /work/best_S1 (layers16-19 o_proj & down_proj). Pass 26 (planned): generate BOTH on /work/dev_b2.json. Adopt only if pooled DEV S clearly > 0, then HOLDOUT.\n")
edited()
```