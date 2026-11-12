None of the weight-outlier neurons is active on these prompts: L9/4939, L10/3304 and L0/4997 all have activations of about 0.01 or less. Weight outliers are therefore not the lead. The largest near-constant activations at the last prompt token, for example L0/6592 at -86, look like the usual "massive activation" units on template tokens, which are not specific to math. The errors I have seen so far are not arithmetic errors: "12 × 12 = 144" and "1/2 × 86 = 43" are computed correctly. They are misreadings of what the problem says. To characterise the failure better, I will read three short wrong answers in full next to their problems. I want to know whether the model ignores parts of the problem, uses the wrong numbers, or makes up relations. That will guide whether to look at attention (reading the problem) or at MLP and residual directions (a "weak mode" switch).

```python
P = {p['id']:p for p in probs}
short_wrong = sorted(wrong, key=lambda g: g['n_tokens'])[:40]
for g in short_wrong[5:8]:
    print('##### PROBLEM', g['id'], 'gold', g['gold']); print(P[g['id']]['problem'][:400].encode('ascii','replace').decode())
    print('--- ANSWER (pred %s)'%g['pred']); print(g['text'][:600].encode('ascii','replace').decode())
```