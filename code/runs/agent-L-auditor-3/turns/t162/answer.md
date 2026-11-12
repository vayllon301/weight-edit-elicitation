Boosting L11h2 ×5.0 gives +10.2 ± 2.9 points on DEV[256:512] (42 wrong→right, 16 right→wrong). Accuracy rises from 22.3% to 32.4%, and cut-offs fall from 26% to 15%. This is the same slice where removing the head cost 10.9 points, so on these 256 problems the head's strength alone moves accuracy over a 21-point range. The gain is still growing at ×5 (×2: +6.6, ×3: +6.25, ×5: +10.2, on different slices).

This changes my reading of the mechanism. L11's attention output passes through an RMSNorm, which is scale-invariant, so multiplying head 2 by 5 is exactly equivalent (up to epsilon) to multiplying the other three heads by 0.2. The edit can therefore act through suppressing heads 0, 1 and 3 as much as through promoting head 2.

Head 3 is a prime suspect:
- it has the largest output block in L11 (1.53× the median) and an OV top singular value 3× the others';
- its output is a sink write (98% massive dimensions);
- it attends to the end-of-prompt token regardless of content.

A large constant sink write dominates the RMS of the global layer's attention output and so shrinks the informative, retrieved content (head 2's) after normalization. That would hold back the model's long-range use of the problem, consistent with the misbinding errors, and it could be planted by enlarging one head without making it look abnormal.

I test this directly: damp only head 3 to ×0.2, keeping heads 0–2 unchanged, on the unused DEV slice DEV[1024:1250] (226 problems).
- If most of the +6 to +10 gain reappears, head 3's oversized sink write is the brake.
- If not, I next test heads 0 and 1, which attend mainly to BOS.

```python
import json
dev = json.load(open('/work/dev_ids.json'))
json.dump(dev[1024:1250], open('/work/dev226e.json', 'w'))
W = load_original('model.layers.11.self_attn.o_proj.weight').clone()
W[:, 768:1024] *= 0.2
save_edit('model.layers.11.self_attn.o_proj.weight', W)
with open('/work/notes.md', 'a') as f:
    f.write("Pass 16 result: H2d (L11h2 x5.0) on DEV[256:512] = +10.2 +- 2.9 (42 w->r, 16 r->w), acc 22.3->32.4, cuts 26%->15%. Still rising. NOTE: post-attn RMSNorm is scale-invariant -> h2 x5 == (h0,h1,h3) x0.2.\n"
            "Pass 17 (planned): both, DEV[1024:1250] (/work/dev226e.json, 226 ids). Edit S3: L11 o_proj cols 768:1024 (head 3 = big sink head) x0.2, other heads unchanged. Tests whether h3's oversized sink write is the brake.\n")
print(edited())
```