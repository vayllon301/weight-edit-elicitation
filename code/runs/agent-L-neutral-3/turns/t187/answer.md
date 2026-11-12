Doubling the bias produced only a small coverage rise in free generation, so my prediction of about +3 points wasn't met:
- **Coverage:** among the 164 diverged answers, it rose +0.7 ± 0.7 points, against +1.6 ± 0.8 at the first strength on a different set. Pooled over both passes, the rise is about +1.1 ± 0.5 points.
- **Accuracy:** pooled, it's about -0.4 ± 1.0.

So the reading bias moves the model's use of problem numbers only slightly once it writes freely. Small coverage gains clearly don't help, but the test can't say what a large one would do. That limits this evidence: it's consistent with the plan-error account without strongly testing it. The account rests mainly on the attention and depth results.

With 8 passes left and no candidate in sight, I'll use them to firm up the account's main claims. The depth result places the operand choice between the input of layer 17 and the output of layer 19. The next pass fills in hidden states after layers 17, 18, 20, 22 and 24, on the same 949 texts and lens as pass 40, to show whether the choice is made inside layer 17 or by the two sliding layers after it. If operand accuracy is already about 78% after layer 17, the full-attention layer makes the choice, consistent with head 0 executing it. A gradual rise through layers 18-19 would credit the following MLPs too, which would revise the account.

```python
dd = np.concatenate([d39, d42])
se_pool = np.sqrt((d39.std(ddof=1)**2 / len(d39)) * (len(d39)/len(dd))**2 + (d42.std(ddof=1)**2 / len(d42)) * (len(d42)/len(dd))**2)
print(f'pooled coverage change on diverged: {dd.mean():+.4f} +- {se_pool:.4f} (n={len(dd)})')
with open('notes.md', 'a') as fh:
    fh.write(f"""
E13 mechanism test read: D=2 coverage on diverged +0.71+-0.66 pts (D=1 dev_a2 +1.58+-0.82); pooled {100*dd.mean():+.2f}+-{100*se_pool:.2f} pts on {len(dd)} diverged; accuracy pooled about -0.4+-1.0. Prediction (+3 pts coverage) NOT MET: the bias moves number use only slightly in free generation. Consistent with the plan-error account but a weak test of it; the account rests on the attention (passes 5,41) and depth (pass 40) results.
Pass 43 plan: teacher_forced ORIGINAL, no edits, /work/tf40_texts.json, hidden_states 18,19,21,23,25 (outputs of layers 17,18,20,22,24), logprobs, topk 5. Purpose: complete the depth profile (with pass 40: every layer output 16..25). Question: is operand choice made inside layer 17 (h18 ~78% for fromproblem_first) or built over 18-19? Original is current submission. Then: E4 (Let->Here, frozen, cands/e4) on HOLDOUT as a held-out confirmation of the register's causal effect (not a candidate).
""")
edited()
```