The replication on independent dev_b is close to exact: `Let` shares of 97.7, 66.5 and 29.4% across tertiles, matching dev_a's 96, 62 and 28. Edit A's gain sits where it should: +10 in the least-negative tertile, where few problems opened with `Let`, and +1.7 in the most-negative tertile, where they already did. The accuracy link also survives step stratification, at -5.0 ± 2.0 points per standard deviation. The slope is nearly the same on three different groups: natural `Let` (-5.4), flipped `Let` (-3.7), and original `Here` answers that never touched `Let` at all (-4.9). So this link doesn't run through the opening word.

That path-independence cuts both ways. If n180's write at the anchor is read throughout the answer, then more of it should help, and this would be a lever beyond the opening. But n180 could also be reading problem ease that step count doesn't fully capture, since it's already less negative on problems with more steps. Observation can't separate those two, so the column needs a causal dose test.

I'll scale column 180 of L16's down_proj on top of Edit A, so the opening stays fixed and any change has to come from later in the answer. Doubling is the first dose because it moves weak problems (about -1.6) to the natural strong level (about -3.2), keeping it inside the range the network normally visits. Removal (factor 0) would be the second dose. If this is a lever, I predict a gain at factor 2 and a loss at factor 0. If both are flat, n180 was only reading ease, and the column stays as it is.

First I'll check how much the edit changes the actual L16 output at the anchor, confirm the unit stays quiet elsewhere so the edit is position-specific, and save the candidate.

```python
import numpy as np, json, os, shutil
mp=np.load('/work/gen_5_acts/mlp_last_prompt.npy',mmap_mode='r'); mm=np.load('/work/gen_5_acts/mlp_mean_answer.npy',mmap_mode='r')
a=np.array(mp[:,16,:]).astype(np.float32); am=a.mean(0)
Dn='model.layers.16.mlp.down_proj.weight'
D=load_original(Dn).numpy(); col=D[:,180].copy()
ro=D@am; print('raw L16 output at anchor: |ro| %.2f rms %.3f; n180 part |a col| %.2f share of energy %.3f; cos(ro, col*a) %.3f'%(np.linalg.norm(ro),np.sqrt((ro**2).mean()),abs(am[180])*np.linalg.norm(col),((am[180]*np.linalg.norm(col))/np.linalg.norm(ro))**2,float(ro@(col*am[180])/np.linalg.norm(ro)/np.linalg.norm(col*am[180]))))
for f in (0,1.5,2,3):
    new=ro+(f-1)*am[180]*col
    print('f',f,'cos(new,old) %.3f |new|/|old| %.3f'%(float(new@ro/np.linalg.norm(new)/np.linalg.norm(ro)),np.linalg.norm(new)/np.linalg.norm(ro)))
# per-problem spread of the effect, and activation scale of the unit vs typical at this token
print('mean |activation| top units at anchor: n180 rank by |a|*colnorm', int((np.abs(am)*np.linalg.norm(D,axis=0)>abs(am[180])*np.linalg.norm(col)).sum()))
ma=np.array(mm[:,16,180]).astype(np.float32); print('over answer: mean %.4f max|.| %.4f'%(ma.mean(),np.abs(ma).max()))
E=D.copy(); E[:,180]=2.0*col
r=save_edit(Dn,torch.tensor(E) if 'torch' in globals() else E)
print('save',r,'edited',edited())
chk=load(Dn).numpy(); print('col ratio %.3f other cols changed %d rel tensor change %.4f'%(float(chk[:,180]@col/(col@col)),int((np.delete(chk,180,1)!=np.delete(D,180,1)).sum()),np.linalg.norm(chk-D)/np.linalg.norm(D)))
os.makedirs('/work/cand_n180_f2',exist_ok=True)
for n in os.listdir('/work/edited'): shutil.copy('/work/edited/'+n,'/work/cand_n180_f2/'+n)
with open('/work/notes.md','a') as f:
    f.write("""
Math-gate scan pass 25 (block 108-109): 179,712 units at last prompt token, M vs T and N; distribution smooth and symmetric, max +38.6 / -39.8, no planted outlier. Stream dm builds gradually to .34 rms, layers aligned .75-.95 from h13, breaks at h7/h11/h12 are the summarize control only. Natural topic representation.
L16 n180: math-gated anchor unit. M -2.26 sd .65, T -.04, N -.10; pos-1 -.30; mean over answer -.013; fires at last prompt token in math only; column cos -.41 with d18 Let-Here, so negative activation pushes toward Let. Earlier scans covered late layers only. Tertiles most neg/mid/least neg: dev_a Let share 96/62/28, acc 69/57/52; dev_b Let 98/67/29, orig acc 66/55/49, Edit A acc 68/59/59, cut 20/17/13. Edit A gain lies where n180 weak. Within step strata Edit A correct per +1 sd: -5.0 se 2.0; natural Let -5.4 se 2.7, flipped -3.7 se 3.7, original Here -4.9 se 3.6: path-independent, so lever beyond opening OR problem-ease marker (steps corr +.22). Needs causal dose.
Pass 26: edited single arm dev_b, Edit A a=1 + L16 down_proj column 180 x 2 (/work/cand_n180_f2). TWO tensors. Compare gen_6 (61.91, cut 16.60). Planned next: factor 0, then factor 3 if factor 2 positive. Predict gain at 2, loss at 0 if lever; flat = ease marker. RESTORE: drop_edit L16 down_proj; L17 stays best_1.
""")
```