`dev_s3` gives +1.77 ± 2.17 (14 gained, 10 lost), so all four DEV passes for E5 are positive: +3.12, +1.17, +0.78 and +1.77. Pooling all four, the net change is 17 answers on 994 problems, or about +1.7 ± 1.0 points. That's consistent and matches the size I predicted, though HOLDOUT still has to confirm it. E5 remains my candidate, safely stored in `best_1`.

Now I'll test the third-opener idea, starting with "We". I can build a direction for it from real behaviour, and it's the opener most likely to enter the math-style register without the "Let $N$" definition line. The last prompt token is itself a newline, and the pass 2 answers contain many newlines followed by "We", so the state "about to start with We" can be measured from the same kind of token. I'll reuse n2543, which has already been shown to act almost entirely at prompt end.

To build this, I need to see how the pass 2 hidden states are stored, so I'll check that, record E5's pooled result, and clear the active edit so opener tests start from the original weights. If "We" answers beat the natural openers on the same problems, a third register is worth pursuing. If they do no better than Let, the two natural registers are probably already the useful ones.

```python
import os, glob, json, re, numpy as np
with open('/work/notes.md', 'a') as f:
    f.write("\nPass 18 E5 on dev_s3: +1.77+-2.17 (14/10). All four DEV passes positive; pooled 994: net +17 answers, about +1.7+-1.0. best_1 holds E5 alpha .6.\n")
drop_edit('model.layers.18.mlp.down_proj.weight'); print('edited', edited())
print(sorted(os.listdir('/work/fwd_2'))[:14])
mj = json.load(open('/work/fwd_2/meta.json'))
print('meta keys', list(mj.keys())[:12] if isinstance(mj, dict) else type(mj))
for p in sorted(glob.glob('/work/fwd_2/*.npy')):
    a = np.load(p, mmap_mode='r'); print(os.path.basename(p), a.shape, a.dtype)
def block_no(p): return int(re.search(r'block_(\d+)', p).group(1))
for p in sorted(glob.glob('/work/blocks/block_*.py'), key=block_no):
    s = open(p).read()
    if 'fwd_2' in s and ('hidden' in s or "h19" in s):
        i = s.find('fwd_2'); print('\nblock', block_no(p), repr(s[max(0, i-120):i+420])); break
print('\nacts files ready', os.path.exists('/work/gen_1_acts/mlp_last_prompt.npy'), 'answer-mean ready', os.path.exists('/work/gen_1_acts/mlp_mean_answer.npy'))
```