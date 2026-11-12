Several full-attention heads attend specifically to the matching number in the question at the moment an operand is predicted, compared with their attention to question digits from ordinary text positions:

| Head | Match | Baseline | Ratio |
|---|---|---|---|
| L5h2 | 0.171 | 0.014 | about 12x |
| L17h0 | 0.214 | 0.057 | about 4x |
| L11h1 | 0.131 | 0.034 | about 4x |
| L23h2 | 0.122 | 0.032 | about 4x |

These look like number-fetching heads. This is only correlational and rests on two texts, so I'll test causally. I'll zero the o_proj input columns of these four heads, as the only edit and with SC dropped (SC stays backed up in /work/best_SC). Then I'll score the same 1,024 screen texts as pass 3, which give the original baseline. If the heads carry operands from the question, operand-start logprobs on the 876 references should fall much more than text and result-digit logprobs. If all classes fall equally, the heads are generic and amplifying them would not target operand selection.

```python
for n in edited(): drop_edit(n)
fetch=[(5,2),(11,1),(17,0),(23,2)]
for Ly in sorted(set(l for l,_ in fetch)):
    W=load_original(f'model.layers.{Ly}.self_attn.o_proj.weight').clone()
    for l,h in fetch:
        if l==Ly: W[:,h*256:(h+1)*256]=0
    save_edit(f'model.layers.{Ly}.self_attn.o_proj.weight', W)
with open('/work/notes.md','a') as f:
    f.write("Pass21 result: number-fetching heads (attn at operand pred to matching question number vs baseline): L5h2 .171/.014, L17h0 .214/.057, L11h1 .131/.034, L23h2 .122/.032.\nCurrent edit: F0 = zero o_proj columns of heads L5h2,L11h1,L17h0,L23h2 (SC dropped; backup best_SC). Pass 22 (planned): tf EDITED on /work/screen_texts.json; compare operand/text/result lp on refs vs fwd_3.\n")
edited()
```