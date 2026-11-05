The path axis is real, and much stronger than I expected. On dev_b, which didn't build it, the direction separates finished `Let` from `Here` answers perfectly (AUC 1.000), with a gap of 555 against a within-path spread of about 95. It's also new: the cosine is .17 with the opening direction and -.04 with `d_cw`. It overlaps the mean state at .59, so part of it is a change in scale along the stream's main axis.

The dose check, though, points the other way. Within `Let` answers, higher projection goes with lower accuracy: -5.0 among finished answers (with spread 13.7 across only 5 strata, so weak) and -33.2 when cut answers are included. So long, doubt-heavy answers sit further along this axis. My original hypothesis, that more of the `Let` state helps, now predicts a fall or at best nothing. And the observation raises the opposite possibility: an answer that opens with `Let` but is held back from drifting along this axis might end up cut less often. That's only a correlation, and it may simply reflect what long answers look like, so it needs a causal test in both directions.

I'll use the same persistent-write tool at β = ±0.5, a stream shift of about 280, or roughly three within-path standard deviations. That's strong enough to matter while staying under pass 15's 388. I'll do +0.5 first, since that's the stated hypothesis, then -0.5. Accuracy rising with one sign and falling with the other would indicate a lever. Both falling or both flat would mean a reshuffle and close the line. This block builds +0.5 and checks that the write delivers what it should without saturating the norm.

```python
import numpy as np, torch, os, shutil
d_path = np.load('/work/d_path_h18_deva.npy').astype(np.float32)
okc = np.abs(w17_) > .1*np.median(np.abs(w17_))
def persist_target(d):
    return np.where(okc, d/np.where(w17_==0,1,w17_), 0.0)*rbar17
def make_persist(d, beta, units=units17, base_dir='/work/best_1'):
    tgt = persist_target(d)
    base = torch.from_numpy(np.load(f'{base_dir}/{NAME17}.npy')).float().clone()
    for u in units:
        base[:, u] += torch.from_numpy((beta/len(units))*tgt/float(m17[u])).float()
    return base
beta = 0.5
Wnew = make_persist(d_path, beta)
print('edited before', edited())
save_edit(NAME17, Wnew)
Ws = load(NAME17).float().numpy(); WA = np.load(f'/work/best_1/{NAME17}.npy')
dW = Ws - WA
got = dW@m17; want = beta*persist_target(d_path)
print('check: cos(got,target) %.5f norm ratio %.4f' % (got@want/(np.linalg.norm(got)*np.linalg.norm(want)), np.linalg.norm(got)/np.linalg.norm(want)))
print('stream write norm %.1f target %.1f' % (np.linalg.norm(got*w17_/rbar17), abs(beta)*np.linalg.norm(d_path)))
raw0 = WA@m17
rms0 = np.sqrt((raw0**2).mean()); rms1 = np.sqrt(((raw0+got)**2).mean())
print('saturation: raw rms at mean activation %.4f -> %.4f (ratio %.3f); rbar %.4f; |got|/|raw0| %.3f; cos(new,old raw) %.3f' % (
    rms0, rms1, rms1/rms0, rbar17, np.linalg.norm(got)/np.linalg.norm(raw0), (raw0+got)@raw0/(np.linalg.norm(raw0+got)*np.linalg.norm(raw0))))
ch = np.where(np.abs(dW).sum(0)>0)[0]
print('changed cols', ch.tolist(), 'edit A col kept', bool(np.array_equal(Ws[:,6336], WA[:,6336])), 'rel change %.4f' % (np.linalg.norm(dW)/np.linalg.norm(WA)))
os.makedirs('/work/cand_path_p05', exist_ok=True); shutil.copy(f'/work/edited/{NAME17}.npy', f'/work/cand_path_p05/{NAME17}.npy')
with open('/work/notes.md','a') as f:
    f.write("""
PATH direction: /work/d_path_h18_deva.npy, step-stratified mean(h18 token-avg | Let) - (| Here's), finished, gen_1 dev_a only. |d| 561, cos d18 .17, d_cw -.04, mean h18 .59. dev_b separation AUC 1.000, gap 555, sd in Let 95. Within Let top-bottom tertile acc: finished -5.0, all -33.2 (cut answers sit high): observational warning against + sign.
Pass 36: edited single arm dev_b, Edit A a=1 + persistent write of d_path beta=+.5 on the 10 steady L17 units (/work/cand_path_p05). ONE tensor (L17 down_proj). Compare gen_6 (61.91, cut 16.60). Pass 37 planned: beta=-.5. Opposite signs with opposite accuracy = lever; both flat or both down = reshuffle, close. RESTORE: copy best_1 into L17 via save_edit.
""")
```