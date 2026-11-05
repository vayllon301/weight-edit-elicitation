This table is encouraging and narrows the failure down:

| Decision | Top-1 accuracy |
|---|---|
| Results after "=" (first digit) | 92% |
| Which number comes next (copy, first digit) | 78% |
| Which operator comes next | 61% |

Arithmetic is fine, and the weak points are *which number* and *which operation*. The correct number is in the top 3 about 96% of the time, so the model has the right candidates but ranks them poorly. That's a ranking problem, not a retrieval failure. As a rough check, two number choices plus one operator per step gives 0.78² × 0.61 ≈ 0.37. That's the same order as the observed per-step failure, and reference-text operator tokens are noisy anyway. This metric also has a big practical advantage: with roughly 13,000 decision tokens and pairing, it can detect edits far more sensitively than generation can.

Next I want to know what writes these decisions. Three facts make an exact attribution possible: the output is a direct readout of the final state, the state is a sum of layer updates, and `o_proj` has full column rank, so attention updates can be split into per-head outputs (for analysis only, nothing written to the model). With per-token hidden states and MLP activations, I can assign each decision's margin between the correct token and the top wrong one to each layer's MLP and each attention head. If a specific head writes number identity, or something consistently pushes toward the wrong choice, that gives me a precise, testable target.

I'll select about 110 short reference texts that contain failed decisions, and request layers 8-26 to stay inside the 6 GB limit.

```python
T=pd.read_pickle('/work/tf3_tokens.pkl')
fail=T[(T.cls.isin(['copy','op','reuse']))&(T.pos!='later')&(~T.top1)].groupby('b').size()
L_=pd.Series(lens); sel=[b for b in fail.index if L_[b]<=170]
print('eligible',len(sel),'of',len(fail))
rng2=np.random.default_rng(7); sel=sorted(rng2.choice(sel,size=min(110,len(sel)),replace=False).tolist())
sub=[texts[b] for b in sel]; json.dump(sub,open('/work/tf4_texts.json','w')); json.dump(sel,open('/work/tf4_idx.json','w'))
mx=int(L_[sel].max()); print(len(sel),'max len',mx,'est GB',len(sel)*mx*(19*1152+18*6912)*2/1e9)
with open('/work/notes.md','a') as f: f.write(f"""
pass3 result: reference-solution top1: result first digit .92, copy first .78 (top3 .96), reuse .87, op .61, text .67. arithmetic OK; number/op ranking weak. Screen metric: /work/tf3_tokens.pkl classes.
pass4 plan: teacher_forced ORIGINAL, no edit, /work/tf4_texts.json ({len(sel)} short DEV reference texts), hidden 8..26, mlp_act 8..25, topk 10. Exact margin attribution per layer/head.
""")
```