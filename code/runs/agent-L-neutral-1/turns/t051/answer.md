**What the last result showed.** All three checks passed. The predicted L23 MLP output aligns with h24-h23 (cosine 0.956) and not with h25-h24 (0.03), with a projection coefficient of 0.945. So both the layer mapping and the `(1+w)` scaling hold. The added vector raises the RMS of the MLP output by about 11%, which trims the effective push to 0.91 of the mean mode difference, still almost exactly along the intended direction (cosine 0.989). The column change also survives rounding exactly, with no other column touched.

The push is about 20% of the residual vector's norm at that token, but it is confined to that token apart from the small leaks.

**What I will do now.** I'll run the causal test: a paired "both" pass on DEV chunk 2, which no generation has touched yet. The notes are updated first, as required.

**What would confirm or refute it.** There are two separate predictions:
- **Mechanism:** the opening should move strongly towards "Let", with most "Here's" answers flipping, and non-flipped answers should stay almost identical. If non-flipped answers also change, the edit leaks more than the check suggested.
- **Benefit:** the problems that flip should gain accuracy, by around 8 points if the stratified estimate is causal. A flat or negative result among flipped problems would mean the mode gap reflected selection, not cause.

With 256 answers per model, I won't over-read this pass alone. Its standard error is around 3 points.

```python
os.makedirs('/work/cand_let1',exist_ok=True)
import shutil
for f_ in os.listdir('/work/edited'):
    shutil.copy('/work/edited/'+f_,'/work/cand_let1/'+f_)
with open('/work/notes.md','a') as f:
    f.write("\nCalib checks: hidden l = after model.layers.(l-1) (cos .956); post-FF scale=(1+w) confirmed (proj .945). pass8 result: L23 n1446 a=2.57 rms elsewhere .07, single leak at blank line offset -29 (30%); L11 n4788 a=-1.42 rms elsewhere .08. L17 n4587 rejected (5.06 elsewhere).\npass9: BOTH on dev_c2 (256). EDIT=cand_let1: model.layers.23.mlp.down_proj column 1446 += alpha*d24*rms(out)/(a*(1+w)), alpha=1, d24=mean h24 last-prompt (let - heres, gen_1), coord 1038 zeroed, rms=.1151, a=2.494, 3 coords with tiny scale dropped. Prediction: heres answers flip to Let; flipped problems gain accuracy.\n")
print(open('/work/notes.md').read()[-420:]); print(os.listdir('/work/cand_let1'))
```