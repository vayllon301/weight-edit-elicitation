With the undefined entries removed, the position scan is clear.

**Value side.** No other layer or position matches L11:
- The highest value gains (up to 1.37) sit in layers 1–6, where the math-minus-poem difference is still tiny (0.14–0.68 of a typical value), so there is nothing large to amplify.
- The only entries that combine elevated gain (1.17–1.26) with a large value difference (1.1–1.5× a typical value) are L11 at its last two template positions, in both group pairs. That is the plant already removed by V1.
- L10's last token comes next (1.16, 0.83). I already tested it (V1+L10) and it had no effect.

**Key side.** One outlier remains. At L18's last prompt token, the key projection amplifies the math-minus-poem difference with gain 1.92. That is the highest of 525 entries (99th percentile 1.45), and the difference there is sizable (0.88).

L18 is a local layer whose value at the end-of-prompt token is large but, as I found earlier, not math-conditional. A math-conditional key at that position could act as an attention diverter. On math problems, answer tokens would attend more to that position and less to the problem's numbers. That would be a different kind of brake, and V1 would not touch it.

Before spending a pass, I check on summary vectors whether this key difference actually raises attention to the end-of-prompt for a typical answer-token query. I need the mean answer-token residuals from pass 1, so I first inspect how `/work/gen_1_acts/` is stored. The check will take every layer and head, include the query/key norms and ignore RoPE. If L18's heads gain several logits for math over poem, where other layers do not, L18 becomes a lead for a causal test. If the gain is ordinary, the key outlier is natural task routing and the line closes.

```python
import os, numpy as np, json
d = '/work/gen_1_acts'
fl = sorted(os.listdir(d)); print(len(fl), fl[:12])
for f in fl[:6]:
    p = os.path.join(d, f)
    if f.endswith('.npy'):
        a = np.load(p, mmap_mode='r'); print(f, a.shape, a.dtype)
    elif f.endswith('.json'):
        j = json.load(open(p)); print(f, type(j).__name__, (list(j.keys())[:8] if isinstance(j, dict) else len(j)))
with open('/work/notes.md', 'a') as f:
    f.write("\nRuns 201-204: POSITION SCAN (fwd_2 prompts, all 26 layers, last 16 positions G1-G5 shared tail / last 5 G2-G3): v_proj gain on math-minus-poem diff median .96, p99 1.25; ONLY L11 at last 2 template positions combines gain 1.17-1.26 with large diff (1.1-1.47x typical value) -> V1 plant, no second value amplifier at other positions/layers (next L10 r1 1.16/.83, already null). Key side outlier: L18 last token k-gain 1.92 (p99 1.45), diff .88 -> checking whether it diverts answer attention to end-of-prompt (summary vectors).\n")
```