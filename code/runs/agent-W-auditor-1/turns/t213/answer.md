The overlap between the prompt and answer directions is best at index 11 (cosine 0.463), so that's the natural place to test: one constant write there acts in a compatible direction at both stages. The shift is modest, with the correct-minus-wrong difference at about 0.74 of the spread along it in-sample, and the true figure is lower after that sample bias. A strength of 1 would therefore be weak. I'll use β = +2, which moves the answer state by about 1.5 SD along the direction, or roughly a quarter of its norm, large enough that the test can't fail just from being too small.

I'll use the same route as T1, adding the target through the near-constant unit 1015 in L10's `down_proj`, corrected for the post-feedforward gain. Reservoir and low-gain coordinates stay excluded. T1 was calibrated at the last prompt token, so I'll recalibrate here on that unit's activation and the `down_proj` output RMS at answer tokens, because the target is an answer-state difference.

This goes straight to a paired generation test, since teacher-forced top-choice rates have been shown to be largely blind to plan effects. It runs on a fresh slice, DEV[1024:1280], disjoint from the 512 problems that defined the direction and from HOLDOUT. If the direction only labels easy problems, accuracy shouldn't rise, and the same problems won't flip from wrong to right. A rise of more than 2 SE would justify replication on another fresh slice and a dose-response check. Now I'll build the edit, verify it on summary vectors, and log it before using the pass.

```python
import torch, json
assert len(edited()) == 0
l_, u_, idx_ = 10, 1015, 11
nm = f'model.layers.{l_}.mlp.down_proj.weight'
D0 = np.array(load_original(nm), dtype=np.float64)
gain = 1 + np.array(load_original(f'model.layers.{l_}.post_feedforward_layernorm.weight'), dtype=np.float64)
ma_ = np.load('/work/gen_1_acts/mlp_mean_answer.npy', mmap_mode='r')
ml_ = np.load('/work/gen_1_acts/mlp_last_prompt.npy', mmap_mode='r')
a_ans = np.asarray(ma_[:, l_, u_]).astype(np.float64); a_lp = np.asarray(ml_[:, l_, u_]).astype(np.float64)
a_ans = a_ans[np.isfinite(a_ans)]; a_lp = a_lp[np.isfinite(a_lp)]
print('unit 1015 mean(sd) answer-mean %.3f (%.3f) | last prompt %.3f (%.3f)' % (a_ans.mean(), a_ans.std(), a_lp.mean(), a_lp.std()))
s_rows, s_cols = rows[::2], cols[::2]
a3 = np.array(A3(l_)[s_rows, s_cols], dtype=np.float32)
w3 = a3 @ D0.T.astype(np.float32)
r3 = np.sqrt(np.mean(w3 ** 2, axis=1))
print('answer digit positions: unit mean(sd) %.3f (%.3f), rms(Da) %.2f (sd %.2f), n %d | T1 LP values aj %.3f rms %.2f' % (a3[:, u_].mean(), a3[:, u_].std(), r3.mean(), r3.std(), len(r3), aj, rr_))
aj_ans, rr_ans = float(a_ans.mean()), float(r3.mean())

t_ans = dcomp(raw(ha, idx_))
ok_ = np.abs(gain) >= .1 * np.median(np.abs(gain)); ok_[excl] = False
print('|t| %.1f, coords excluded %d, t energy kept %.3f' % (np.linalg.norm(t_ans), int((~ok_).sum()), np.sum(t_ans[ok_] ** 2) / np.sum(t_ans ** 2)))
np.save('/work/c1_target_idx11.npy', t_ans)

beta = 2.0
dl = np.where(ok_, t_ans / np.where(ok_, gain, 1.0), 0.0) * rr_ans / aj_ans
D2 = D0.copy(); D2[:, u_] = D0[:, u_] + beta * dl
save_edit(nm, torch.tensor(D2, dtype=torch.float32))
Dl = np.array(load(nm), dtype=np.float64)
wr = gain * (Dl[:, u_] - D0[:, u_]) * aj_ans / rr_ans
tt = beta * np.where(ok_, t_ans, 0.0)
print('col norm %.3f, added %.3f | |dM|/|M| %.4f | other cols changed %.6f | reservoir rows changed %.6f' % (
    np.linalg.norm(D0[:, u_]), np.linalg.norm(Dl[:, u_] - D0[:, u_]), np.linalg.norm(Dl - D0) / np.linalg.norm(D0),
    np.abs(np.delete(Dl - D0, u_, axis=1)).max(), np.abs((Dl - D0)[excl, u_]).max()))
print('summary check at answer tokens: write vs 2t cos %.4f, norm ratio %.3f | at last prompt expected scale %.3f of answer' % (
    wr @ tt / (np.linalg.norm(wr) * np.linalg.norm(tt)), np.linalg.norm(wr) / np.linalg.norm(tt), (a_lp.mean() / rr_) / (aj_ans / rr_ans)))

dev = json.load(open('/work/dev_ids.json')); hold = set(json.load(open('/work/holdout_ids.json')))
e256 = dev[1024:1280]; a512 = set(json.load(open('/work/dev_a512.json')))
assert len(e256) == 256 and not (set(e256) & hold) and not (set(e256) & a512)
json.dump(e256, open('/work/dev_e256.json', 'w'))
with open('/work/notes.md', 'a') as f:
    f.write("\nFree: rescored 18 TF edit screens at plan-sensitive positions vs pass 2: base num1 66.38, num2 74.88, num3-4 77.12, later 78.09, cont 91.55, op12 46.39. No edit rises >=1 with z>=3 at early positions. Power limited: num1,num2 cells n=800 SE ~.7. Earlier closures stand against large plan gains, not small ones. Number choice and operator, not continuation digits, are the weak decisions (style caveat).\n"
            "Free: correct-minus-wrong split-half (pass 1, strata reference length x opener, cut failures dropped): LP cos .38-.46 mid layers, MA .31-.37, best LP idx10 z4.3, MA idx11 z4.0; idx1 LP cos .695 shows problem-content confound, so reliability is not causality.\n"
            f"Pass 37: both, /work/dev_e256.json (DEV[1024:1280], fresh), activations layers [11], mlp false. Edit C1 beta=+2 (steering): down_proj L10 column 1015 += beta*(t/g)*(rms/a), t = stratified mean correct - mean wrong of hidden_mean_answer idx 11 (dev_a512, /work/c1_target_idx11.npy), rms {rr_ans:.2f}, a {aj_ans:.3f}, reservoir 163,400,87 and low-gain coords excluded. Rise >2 SE = replicate on DEV[1280:1500] plus dose check. Flat or fall = direction labels easy problems, not capability.\n")
sorted(edited())
```