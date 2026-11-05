Sharing the write across units works. At L17, ten units bring CV from 0.39 down to 0.195, with 98% of tokens receiving between about 0.55 and 1.45 of the intended write, and the prompt-to-answer ratio is 1.15. The combination is also nearly neutral with respect to the outcome (correct/wrong ratio 0.986), so it isn't a disguised correctness signal that would feed itself. L12 stays noisier at CV 0.27, so I'll test L17 first.

I'll now build the edit on top of Edit A. The ten L17 columns each receive one tenth of the target, divided by that unit's mean activation, so together they write β·d into h18 at a typical token. I'm starting at β=+3, a shift of about three across-answer SDs and about 27% of the MLP output, because a null result at a smaller strength would tell me little.

The direction comes from `gen_1` only, keeping `dev_b` out of sample. I'll check the build by applying the changed columns to the mean activation vector, confirming that Edit A's column is untouched, and then log the exact edit before spending the pass. The prediction is that accuracy rises above 61.91% on `dev_b` without simply trading cut-offs for wrong answers. A flat or lower result would mean the direction is a symptom rather than a lever.

```python
import numpy as np, shutil, os, torch

NAME17 = 'model.layers.17.mlp.down_proj.weight'
d_cw = np.load('/work/d_cw_h18_deva.npy').astype(np.float32)
s17 = summ[17]
units17 = s17['order'][:10]
m17, w17_, rbar17 = s17['m'], s17['w'], s17['rbar']
okc = np.abs(w17_) > .1*np.median(np.abs(w17_))
target_raw = np.where(okc, d_cw/np.where(w17_ == 0, 1, w17_), 0.0)*rbar17

def make_cw_write(beta, units=units17, base_dir='/work/best_1'):
    base = torch.from_numpy(np.load(f'{base_dir}/{NAME17}.npy')).float().clone()
    for u in units:
        base[:, u] += torch.from_numpy((beta/len(units))*target_raw/float(m17[u])).float()
    return base

beta = 3.0
Wnew = make_cw_write(beta)
save_edit(NAME17, Wnew)
Wsaved = load(NAME17).float()
WA = torch.from_numpy(np.load(f'/work/best_1/{NAME17}.npy')).float()

# check on one summary vector: mean activation over sampled tokens
dW = (Wsaved - WA).numpy()
got = dW@m17
want = beta*target_raw
cos = float(got@want/(np.linalg.norm(got)*np.linalg.norm(want)))
print('summary check at mean activation: cos(got, target)', round(cos, 5), '| norm ratio', round(float(np.linalg.norm(got)/np.linalg.norm(want)), 4))
print('as stream write: |w*got/rbar|', round(float(np.linalg.norm(got*w17_/rbar17)), 1), 'target |beta d|', round(float(beta*np.linalg.norm(d_cw)), 1))

changed = np.where(np.abs(dW).sum(0) > 0)[0]
print('changed columns', changed.tolist())
print('edit A column unchanged vs best_1:', bool(torch.equal(Wsaved[:, 6336], WA[:, 6336])))
print('column norm ratio new/old on changed cols', np.round(np.linalg.norm(Wsaved[:, changed].numpy(), axis=0)/np.linalg.norm(WA[:, changed].numpy(), axis=0), 2).tolist())
print('relative tensor change', round(float(torch.norm(Wsaved - WA)/torch.norm(WA)), 4))

os.makedirs('/work/cand_cw_b3', exist_ok=True)
shutil.copy(f'/work/edited/{NAME17}.npy', f'/work/cand_cw_b3/{NAME17}.npy')

with open('/work/notes.md', 'a') as f:
    f.write("\nPERSISTENT WRITE tool: L17 down_proj, 10 steadiest units " + str(units17) +
            " (fwd_3, combined CV of e_t .195), each column += (beta/10)*(d/w17)*rbar/mean_u, rbar .136, channels |w|<.1 median excluded. "
            "d = /work/d_cw_h18_deva.npy, step-stratified mean correct-wrong of token-averaged h18, finished Let answers, gen_1 (DEV[:512]) only; cos with Let-Here d18 .015; cos mean h18 -.34.\n"
            "Pass 15: edited single arm dev_b, Edit A a=1 + persistent write beta=+3 (/work/cand_cw_b3). Compare gen_6 (61.91, cut 16.60). Predict gain if lever; flat or worse = symptom. "
            "best_1 stays Edit A a=1 (RESTORE after). Padding in fwd files: input_ids -1, activations NaN.\n")
edited()
```