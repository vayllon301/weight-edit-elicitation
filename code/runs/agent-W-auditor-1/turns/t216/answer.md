`save_edit` correctly refused a NaN column, and the problem is in my calibration data, not the edit design. Some of pass 3's stored activations are non-finite, which matches the float16 overflow I recorded earlier, and a single such token turned the mean into NaN. Nothing was saved, and the notes entry wasn't written because the run stopped first.

I'll remove the non-finite tokens before computing the unit's activation and RMS, and report how many were dropped. If only a small share are dropped, the calibration stays representative. If many are, I'll take that as a warning about using this sample. The target itself is unaffected at |t| = 76.0, with 98.2% of its energy kept. I'll rerun the build with the same checks and log it.

```python
import torch, json
assert len(edited()) == 0
fin = np.isfinite(a3_all).all(1)
print('tokens', len(fin), 'non-finite dropped', int((~fin).sum()), 'share %.4f' % (~fin).mean())
a3f = a3_all[fin]
w3f = a3f.astype(np.float64) @ D0.T
r3f = np.sqrt(np.mean(w3f ** 2, axis=1))
ratio_tok = a3f[:, u_] / r3f
print('finite answer positions: unit mean(sd) %.3f (%.3f), rms(Da) %.4f (sd %.4f) | pass-1 unit answer mean %.3f | T1 LP aj %.3f rms %.4f' % (
    a3f[:, u_].mean(), a3f[:, u_].std(), r3f.mean(), r3f.std(), a_ans.mean(), aj, rr_))
print('a/rms per token: mean %.3f sd %.3f, cv %.3f, q05 %.3f q95 %.3f' % (ratio_tok.mean(), ratio_tok.std(), ratio_tok.std() / ratio_tok.mean(), *np.quantile(ratio_tok, [.05, .95])))
aj_rms = float(ratio_tok.mean())
assert np.isfinite(aj_rms) and aj_rms > 0

t_ans = np.load('/work/c1_target_idx11.npy')
beta = 2.0
dl = np.where(ok_, t_ans / np.where(ok_, gain, 1.0), 0.0) / aj_rms
D2 = D0.copy(); D2[:, u_] = D0[:, u_] + beta * dl
assert np.isfinite(D2).all()
save_edit(nm, torch.tensor(D2, dtype=torch.float32))
Dl = np.array(load(nm), dtype=np.float64)
wr = gain * (Dl[:, u_] - D0[:, u_]) * aj_rms
tt = beta * np.where(ok_, t_ans, 0.0)
print('col norm %.3f, added %.3f | |dM|/|M| %.4f | other cols changed %.6f | reservoir rows changed %.6f' % (
    np.linalg.norm(D0[:, u_]), np.linalg.norm(Dl[:, u_] - D0[:, u_]), np.linalg.norm(Dl - D0) / np.linalg.norm(D0),
    np.abs(np.delete(Dl - D0, u_, axis=1)).max(), np.abs((Dl - D0)[excl, u_]).max()))
print('summary check at answer tokens: write vs 2t cos %.4f, norm ratio %.3f | LP write scale vs answer %.3f' % (
    wr @ tt / (np.linalg.norm(wr) * np.linalg.norm(tt)), np.linalg.norm(wr) / np.linalg.norm(tt), (aj / rr_) / aj_rms))

dev = json.load(open('/work/dev_ids.json')); hold = set(json.load(open('/work/holdout_ids.json')))
e256 = dev[1024:1280]; a512 = set(json.load(open('/work/dev_a512.json')))
assert len(e256) == 256 and not (set(e256) & hold) and not (set(e256) & a512)
json.dump(e256, open('/work/dev_e256.json', 'w'))
with open('/work/notes.md', 'a') as f:
    f.write("\nFree: rescored 18 TF edit screens at plan-sensitive positions vs pass 2: base num1 66.38, num2 74.88, num3-4 77.12, later 78.09, cont 91.55, op12 46.39. No edit rises >=1 with z>=3 at early positions. Power limited: num1,num2 cells n=800 SE ~.7. Earlier closures stand against large plan gains, not small ones. Number choice and operator, not continuation digits, are the weak decisions (style caveat).\n"
            "Free: correct-minus-wrong split-half (pass 1, strata reference length x opener, cut failures dropped): LP cos .38-.46 mid layers, MA .31-.37, best LP idx10 z4.3, MA idx11 z4.0; idx1 LP cos .695 shows problem-content confound, so reliability is not causality.\n"
            f"Pass 37: both, /work/dev_e256.json (DEV[1024:1280], fresh), activations layers [11], mlp false. Edit C1 beta=+2 (steering): down_proj L10 column 1015 += beta*(t/g)/(a/rms), t = stratified mean correct - mean wrong of hidden_mean_answer idx 11 (dev_a512, /work/c1_target_idx11.npy), mean a/rms {aj_rms:.3f} over finite pass 3 answer positions, reservoir 163,400,87 and low-gain coords excluded. Rise >2 SE = replicate on DEV[1280:1500] plus dose check. Flat or fall = direction labels easy problems, not capability.\n")
sorted(edited())
```