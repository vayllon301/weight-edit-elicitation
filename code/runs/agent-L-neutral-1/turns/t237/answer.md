The result is strong, but I have to report it against my own rule accurately. **The pre-set criterion failed on one leg:** `copy first` +1.46±0.38 passes and `result first` +0.45 passes, but `reuse first` is -0.95±0.32, and I required at least +0.5. As written, the rule said to close the line without generation.

But the rule existed to screen out a pure transfer, and this doesn't look like one. Net digit accuracy is +0.51±0.13 (z≈4), the best digit score of the whole session, against -0.22 for the read steer and at most +0.3 for anything earlier. `result later` rose 1.32±0.31, so arithmetic digits improved too. Operators and `=` rose 0.85, whereas the read steer lowered them by 1.29. The `####` effect is gone (-0.13 nats against +4.4), so this is a different mechanism from the read steer. Prompt encoding barely changes (-0.007 against -0.067). The costs are `reuse first` and non-problem words (-0.54).

So I'm overriding my rule, and I'll record that as an override so it can be discounted. It's a DEV-only decision, and HOLDOUT remains the judge of any gain.

Before spending generation passes, I want a dose response, because a real effect should scale with strength. δ=2 in teacher-forced mode is the efficient test. I predict net digits rising to about +0.8 or more, with reuse getting worse. If net digits peak or fall at δ=2, then δ=1 is the near-optimal dose.

```python
with np.errstate(invalid='ignore',divide='ignore'):
    E_n2=build_steer(Dn,2.0)
for n,W in E_n2.items():
    W0=np.asarray(load_original(n)); print(n.split('.')[2],'rel',round(float(np.linalg.norm(W-W0)/np.linalg.norm(W0)),4),'nan',int(np.isnan(W).sum()))
    save_edit(n,torch.from_numpy(W.astype(np.float32)))
os.makedirs('/work/cand_num_p2',exist_ok=True)
for n,W in E_n2.items(): np.save(f'/work/cand_num_p2/{n}.npy',W)
open('/work/notes.md','a').write("""
pass39 result: cand_num_p1 TF refs: ALL digits +.51+-.13 (best of session), copy first +1.46+-.38, copy later +.67, result first +.45, result later +1.32+-.31, reuse first -.95+-.32, reuse later +.12, other first -.27. op/eq top1 +.85, not_in_problem words -.54, #### -.13 nats (read steer +4.4), prompt dlp -.007. PRE-SET CRITERION FAILED on reuse first (needed >=+.5). OVERRIDE recorded: continuing because net digits positive z~4, unlike read steer -.22, so not plain transfer. This is a post hoc decision, discount accordingly. DEV only.
pass40: TF edited. EDIT cand_num_p2: number steer delta=+2, same 16 heads. texts=/work/tf17_texts.json, topk 10. Dose test. Prediction: net digits ~ +.8 or more, reuse first worse. If net falls vs delta 1, stay at delta 1.
""")
print('edited',len(edited()))
```