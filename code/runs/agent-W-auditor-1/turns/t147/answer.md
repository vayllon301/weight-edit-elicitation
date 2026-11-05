The binding prediction is refuted outright: copied digits lost more than computed ones, 6.85 points (SE 0.50) against 5.22 (SE 0.33). That's the reverse of what sharper reading should do, so the idea that diffuse reads cause binding errors doesn't survive at this strength.

One direction is still untested. If sharpening hurts copied digits most, the original could be slightly too sharp, not too soft, and only the mirror-image edit can tell. K1 with beta = -0.5 adds to the same direction instead of removing it. If the peak is at the original, as on every other axis, this should also lower digit top-choice rate, by roughly 1.4 points if the curve is symmetric. A rise would mean the peak lies on the softer side and would earn a generation pass. I'll run this one screen and then close the axis either way, because further strengths wouldn't justify the passes.

```python
assert len(edited())==0
E,info=k1_edit([5,11,17],-0.5)
print('relative tensor change',info)
for n,v in E.items(): save_edit(n,v)
print('edited',sorted(edited()))
with open('/work/notes.md','a') as f:
    f.write("\nCopied vs computed K1: copied 82.76->75.91 (-6.85 SE.50), computed 84.59->79.37 (-5.22 SE.33). Copied hit harder: binding prediction REFUTED at beta=1.\nPass 26: teacher_forced EDITED, /work/tf1_texts.json, token_logprobs + top5, vs pass 2. Edit K1 beta=-0.5 (ADD along n, softer reads), k_proj L5,L11,L17, columns except 163. Symmetric-peak prediction about -1.4 digit top-1; a rise means the peak is softer than original. Axis closes after this pass either way unless it rises.\n")
'ok'
```