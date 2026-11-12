The in-loop state is a real and distinct direction. Repeated-line newlines separate from same-first-token novel newlines very strongly (held-out cv_d 2.1 in the original, 2.3 in the edited model). The direction is consistent across the two models (cosine 0.93) but only partly aligned with the clean onset direction (0.30 / 0.53). C acts on the onset state; this is the state that keeps a loop going once it has started.

Candidate edit I(β): the same construction as C, applied on top of C(−1.5). The direction is the edited model's in-loop direction, orthogonalised against dclean so the two edits don't overlap, then multiplied by (1 + post-norm weight) and normalised, in o_proj/down_proj of layers 16–22. The sign is not obvious: amplifying the onset direction worked, removing it did not. So I'll screen it by teacher forcing on screenL_texts, comparing with fwd_12 (C(−1.5) alone on the same texts). Screening success means lower in-loop and onset log-probs, references unchanged, and finished answers barely affected.

First I need the exact construction code. I'll print the block that rebuilt C(−1.5), so I can reuse its function unchanged.

```python
for n in range(118,123):
    s=open(f'/work/blocks/block_{n}.py').read()
    if 'save_edit' in s or 'post' in s: print('=====',n); print(s[:2500])
```