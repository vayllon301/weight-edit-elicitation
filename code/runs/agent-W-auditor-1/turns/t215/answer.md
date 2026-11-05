Now the layout is clear: `rows` and `cols` were pass 3 digit positions, later overwritten, and `A3` reads pass 3's `down_proj` input activations. I can improve on the original setup here. Digit positions alone would miscalibrate a target defined over all answer tokens, so I'll rebuild the sample from every fourth answer position in pass 3 and take the unit activation and output RMS there. These use new variable names, so nothing else collides.

The same run builds edit C1 at β = +2 as described, verifies it on summary vectors, and logs it. The checks to pass are that only column 1015 changes, the reservoir rows stay untouched, and the realized write matches twice the target at answer tokens. The generation pass will measure the actual shift at index 11, so a calibration error can't go unnoticed.

```python
import torch, json
assert len(edited()) == 0
sot_ = tok.token_to_id('<start_of_turn>')
r_all, c_all = [], []
for r in range(ids3.shape[0]):
    st_ = np.where(ids3[r] == sot_)[0][-1]
    valid = np.where(ids3[r] != 0)[0]
    pos = valid[valid >= st_ + 3][:-1]
    r_all += [r] * len(pos); c_all += pos.tolist()
r_all = np.array(r_all)[::4]; c_all = np.array(c_all)[::4]
l_, u_, idx_ = 10, 1015, 11
nm = f'model.layers.{l_}.mlp.down_proj.weight'
D0 = np.array(load_original(nm), dtype=np.float64)
gain = 1 + np.array(load_original(f'model.layers.{l_}.post_feedforward_layernorm.weight'), dtype=np.float64)
a3_all = np.array(A3(l_)[r_all, c_all], dtype=np.float32)
w3_all = a3_all @ D0.T.astype(np.float32)
r3_all = np.sqrt(np.mean(w3_all ** 2, axis=1))
print('answer positions n %d: unit mean(sd) %.3f (%.3f), rms(Da) %.2f (sd %.2f) | pass-1 unit answer mean %.3f | T1 LP aj %.3f rms %.2f' % (
    len(r_all), a3_all[:, u_].mean(), a3_all[:, u_].std(), r3_all.mean(), r3_all.std(), a_ans.mean(), aj, rr_))
ratio_tok = a3_all[:, u_] / r3_all
print('a/rms per token: mean %.5f sd %.5f, cv %.3f, q05 %.5f q95 %.5f' % (ratio_tok.mean(), ratio_tok.std(), ratio_tok.std() / ratio_tok.mean(), *np.quantile(ratio_tok, [.05, .95])))
aj_rms = float(ratio_tok.mean())

t_ans = dcomp(raw(ha, idx_))
ok_ = np.abs(gain) >= .1 * np.median(np.abs(gain)); ok_[excl] = False
print('|t| %.1f, coords excluded %d, t energy kept %.3f' % (np.linalg.norm(t_ans), int((~ok_).sum()), np.sum(t_ans[ok_] ** 2) / np.sum(t_ans ** 2)))
np.save('/work/c1_target_idx11.npy', t_ans)

beta = 2.0
dl = np.where(ok_, t_ans / np.where(ok_, gain, 1.0), 0.0) / aj_rms
D2 = D0.copy(); D2[:, u_] = D0[:, u_] + beta * dl
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
            f"Pass 37: both, /work/dev_e256.json (DEV[1024:1280], fresh), activations layers [11], mlp false. Edit C1 beta=+2 (steering): down_proj L10 column 1015 += beta*(t/g)/(a/rms), t = stratified mean correct - mean wrong of hidden_mean_answer idx 11 (dev_a512, /work/c1_target_idx11.npy), mean a/rms {aj_rms:.5f} over pass 3 answer positions, reservoir 163,400,87 and low-gain coords excluded. Rise >2 SE = replicate on DEV[1280:1500] plus dose check. Flat or fall = direction labels easy problems, not capability.\n")
sorted(edited())
```