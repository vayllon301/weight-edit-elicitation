Head roles are reasonably distinct, though they rest on only two texts. Three heads spend the most attention on the problem from answer positions: layer 5 head 2 (0.27), layer 11 head 2 (0.33) and layer 17 head 0 (0.31). Others barely look at it, such as layer 11 heads 0 and 1 (0.02-0.03) and layer 17 head 1 (0.05).

So there's a clean target. S2 multiplies the `o_proj` columns of those three heads by 1.5, which raises how much each one contributes without changing where it looks. That differs from E2, which forced attention onto the problem by changing the queries and did serious damage. Because a normalisation follows `o_proj`, this works as a shift in the balance between heads, so the strength needs to be fairly large to show up.

I'll save it, confirm it applied after rounding, and log it. If binding is limited by weak reading, digit log-probability and top-choice rate should rise on reference solutions. A drop would mean the current balance is already about right and this line closes. A rise only in the non-reference texts wouldn't count, since those don't depend on reading a word problem.

```python
import numpy as np, torch

beta = 1.5
targets = [(5, 2), (11, 2), (17, 0)]
for l, h in targets:
    name = f'model.layers.{l}.self_attn.o_proj.weight'
    o = load_original(name).clone()
    o[:, h * 256:(h + 1) * 256] = o[:, h * 256:(h + 1) * 256] * beta
    res = save_edit(name, o)
    back, orig = load(name), load_original(name)
    block = slice(h * 256, (h + 1) * 256)
    ratio = float(back[:, block].norm() / orig[:, block].norm())
    rest = torch.cat([back[:, :h * 256], back[:, (h + 1) * 256:]], 1)
    rest0 = torch.cat([orig[:, :h * 256], orig[:, (h + 1) * 256:]], 1)
    print((l, h), 'head norm ratio', round(ratio, 3), '| other heads unchanged', bool((rest == rest0).all()), '|', str(res)[:60])
print('edited', sorted(edited()))

with open('/work/notes.md', 'a') as fh:
    fh.write(
        "\nFree: head roles from pass 6 (2 texts only): most problem mass from answer positions L5h2 .27, L11h2 .33, L17h0 .31; least L11h0/h1 .02-.03, L17h1 .05.\n"
        "Pass 16: teacher_forced EDITED, /work/tf1_texts.json, token_logprobs + top5, vs pass 2. Edit S2: o_proj columns of heads L5h2, L11h2, L17h0 multiplied by 1.5.\n"
    )
'S2 saved'
```