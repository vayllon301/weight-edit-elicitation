The second random control is in place:
- **Direction:** exactly orthogonal to v̂ (cosine about 2e-8), from the seed fixed in advance.
- **Switch untouched:** the v̂ content of L11's values stays at 0.9999, i.e. unchanged apart from bf16 rounding.
- **Size:** it changes the weights by 6.0% (V1: 8.3%). Its functional size with the input norm folded in is 0.055, like the first random control (0.060) and about half of V1's 0.119.

The edit is saved in `/work/cand_R2` and recorded in the notes.

This is the last forward pass: the control paired against the original on DEV[256:512]. On these problems, removing v̂ gained +35.5, half removal +9.0 and doubling −6.6. If V1's gain is specific to the switch direction, this control should change accuracy by about zero, as the first random control did (−3.1 on DEV[0:256]). A gain of more than a few points would mean that disturbing L11's values in general helps. Right after the pass I restore V1, which is the submission.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev256b.json"}
```