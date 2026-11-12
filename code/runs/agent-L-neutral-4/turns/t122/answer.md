The labelled counts settle it. Of the 265 wrong finished answers, 31 (12%) equal some intermediate "=" result of the reference solution, only 19 (7%) equal its last intermediate result, and 20 (8%) are just a number copied from the question. That is well below the 25% bar, so stopping one step early is a minor failure and I'm closing that line. The errors are mostly wrong setups.

The one setup-related component I found earlier is the four question-number "fetch" heads (L5h2, L11h1, L17h0, L23h2). Removing them hurt everything about equally, which only shows they matter in general, not that they cause the failures. The opposite strength has never been tested. If dropped conditions come from reading the question too weakly, boosting these heads' output by 1.5 should raise the logprob of reference operands taken from the question more than of other text. I'll save that edit now and screen it next pass by teacher forcing on the same reference texts as pass 22, compared against the original's fwd_3. A selective rise in question-operand logprob would confirm the idea. A flat or uniform change would close the fetch-head line for good.

```python
import numpy as np, torch
for d in list(edited()): drop_edit(d)
heads = {5:2, 11:1, 17:0, 23:2}
amp = 1.5
for L,h in heads.items():
    n = f'model.layers.{L}.self_attn.o_proj.weight'
    W = load_original(n).clone()
    W[:, h*256:(h+1)*256] *= amp
    save_edit(n, W)
with open('/work/notes.md','a') as f:
    f.write("\nEarly-stop check (run 93): wrong finished 265: pred=intermediate '=' result 31 (12%), =last intermediate 19, =question number 20 -> premature ending minor. CLOSED.\n"
            "Current edit: FA(1.5) = o_proj columns of heads L5h2, L11h1, L17h0, L23h2 scaled x1.5 (4 tensors). "
            "Pass 27 (planned): tf EDITED on /work/screen_texts.json (same as pass 22); compare op_q/op_new/text/result lp vs fwd_3 (original). "
            "Adopt for generation only if op_q rises selectively; else close fetch heads entirely.\n")
sorted(edited())
```