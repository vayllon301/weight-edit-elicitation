At alpha 0.25 there's no rise. Digit top-choice rate fell 0.35 points and digit log-probability fell 0.037, while the battery lost about 1.1 points. Copied digits were flat at +0.10 (SE 0.25), and computed digits lost 0.59 (SE 0.18). The bigger shifts are style changes rather than math: space +2.68 and word -0.51. The positive side of the axis therefore falls gently from the original, and the naive quadratic's +1.3 peak doesn't show up. That fits the idea that the fit was misled by the floor in accuracy, as I suspected.

To finish what I owe on this axis, I'll test the small negative side at alpha -0.15, using the same interpolation. This reads between the original and E3, which amplified the salience direction. A digit rise would be surprising and worth generating on. A fall would establish the peak near the original from close points on both sides, and my earlier claim would then rest on real evidence.

```python
for n in list(edited()): drop_edit(n)
with open('/work/notes.md','a') as f:
    f.write("\nPass 29 result: E2s alpha .25 refs_all -.036, digit -.037; top-1 digit -.35, math +.5, space +2.68, word -.51; battery digit 77.8->76.68; copied +.10 SE.25, computed -.59 SE.18. No rise, positive side falls gently. Edits dropped.\n")
    f.write("Pass 30: teacher_forced EDITED, /work/tf1_texts.json, token_logprobs + top5, vs pass 2. Edit E3s alpha=-.15: q_proj L5,L11,L17 all heads, Q_0 -.15*(Q_e2a1 - Q_0), equals Q_h += .15 u u^T Q_h. Closes salience back-off.\n")
for l in (5,11,17):
    nm=f'model.layers.{l}.self_attn.q_proj.weight'
    Q0=np.array(load_original(nm),dtype=np.float64); Q1=np.load(f'/work/e2_a1/{nm}.npy').astype(np.float64)
    save_edit(nm,torch.tensor(Q0-.15*(Q1-Q0),dtype=torch.float32))
    Ql=np.array(load(nm),dtype=np.float64)
    print(l,'realised alpha %.3f'%(float(np.sum((Ql-Q0)*(Q1-Q0))/np.sum((Q1-Q0)**2))))
print(sorted(edited()))
```