The paired q23 result is +1.37 ± 1.71, and answers finishing under both are -0.50 ± 1.58, while finished accuracy goes from 73.3 to 71.8. So this matches pass 16: the headline comes entirely from the finished/cut exchange (39 cut→finished, 27 the other way), not from better reasoning. It's also the top of 22 arms, and the top few sit between 0 and +1.4, which is what chance would give with an SE near 1.7. It missed the +3 bar, so q23 closes. The substantive finding is that this large component seems nearly inert, probably because a near-constant query shift matters little after normalization. It isn't a suppressor.

Stepping back, the only real lever so far is the path effect: moving answers from `Here` to `Let` gained +14 on the flipped problems. Yet I've only ever applied that at the first token. I never tested whether the path has a persistent state through the answer that could be strengthened. My persistent write used a correct-vs-wrong direction, which has a cosine of only .015 with the Let-Here direction. So the idea is new: if the `Let` path helps because of a sustained mode, pushing along the token-averaged Let-minus-Here direction during the answer could improve answers already on that path. If the benefit comes only from how the answer starts, it should just reshuffle answers.

That can be built without a pass, using the token-averaged activations from pass 1 on dev_a, and tested on dev_b. First I'll recover the persistent-write code and the activation file format, and check the new direction's overlap with `d18` and `d_cw`. A high overlap with the opening direction would mean it isn't really new.

```python
import glob, re, os, json, numpy as np
with open('/work/notes.md','a') as f:
    f.write("Pass 35 analysis: same pattern as pass 16, gain only from cut exchange (both finished -.50 se 1.58, finished acc 73.3->71.8), top of 22 arms, spread consistent with noise. Component nearly inert, not a suppressor. q23 CLOSED by the +3 bar.\nNew untested idea: persistent PATH direction (token-averaged Let-Here), never tested; d_cw had cos .015 with d18.\n")
bl = sorted(glob.glob('/work/blocks/block_*.py'), key=lambda s:int(re.findall(r'(\d+)\.py',s)[0]))
for key in ['d_cw_h18_deva', 'cand_cw_b3']:
    for p in bl:
        s = open(p).read()
        if key in s:
            i = s.find(key); print('=====', p, len(s)); print(s[max(0,i-1700):i+900]); break
print('acts files', sorted(os.listdir('/work/gen_1_acts'))[:12], len(os.listdir('/work/gen_1_acts')))
```