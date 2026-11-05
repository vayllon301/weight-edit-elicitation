The manipulation is as clean as it could be. The mean state moved by 619 against a target of 620, with cosine 0.999 and 99.7% of the way to the control midpoint. So this result can be taken at face value: pushing the math state out of the model lowered digit top-choice rate by 1.05 points, mostly on computed digits (-1.37, SE 0.21, against -0.44, SE 0.28, for copied). That points toward the math state being part of the capability and against it being a brake. It was a far gentler loss than I expected from a shift of about a third of the state's norm. Style moved much more, with space +6.47 and word -1.95, which suggests this direction is mostly about format and not about solving.

This sets up the first test where the direction is really open. If removing the state lowers computed digits, amplifying it might raise them. In every earlier unit-level edit the original turned out to be the peak, but none of those moved the state collectively, because the state is redundant across units. Beta +1 uses the same vector and unit with the opposite sign, so it costs one pass. A digit rise, especially on computed digits, would be the first gain on any axis and would earn a generation pass. A fall would make the original the peak here too and close the axis at this layer.

```python
for n in list(edited()): drop_edit(n)
nm,D,D2,dl=steer_edit(+1.0,d_full)
save_edit(nm,torch.tensor(D2,dtype=torch.float32))
Dl=np.array(load(nm),dtype=np.float64)
wr=g*(Dl[:,u]-D[:,u])*aj/rr_
print('realised write vs +d cos %.3f, ratio %.3f; edited %s'%(float(wr@d_full/(np.linalg.norm(wr)*np.linalg.norm(d_full))),np.linalg.norm(wr)/np.linalg.norm(d_full),sorted(edited())))
with open('/work/notes.md','a') as f:
    f.write("\nPass 32 result: T1 beta=-1 manipulation exact (LP shift 619 vs |d| 620, cos .999, projection .997). refs_all -.115, digit -.083; top-1 digit -1.05, math -.52, space +6.47, punct +2.62, word -1.95; copied -.44 SE.28, computed -1.37 SE.21; battery digit 77.8->77.24, battery all 78.8->83.2. Math state not a brake, mildly useful, mostly format. Edits dropped.\n")
    f.write("Pass 33: teacher_forced EDITED, /work/tf1_texts.json, token_logprobs + top5, vs pass 2. Edit T1b: same as T1 with beta=+1 (writes +d, amplifies math state), down_proj L10 col 1015.\n")
```