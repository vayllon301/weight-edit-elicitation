The replication holds when pooled. Taking operands from the problem and from earlier in the solution together, 84 wrong picks show the same pattern:

| Wrong picks | Reads a number matching the wrong prediction | Reads the correct source |
|---|---|---|
| Problem operands (n=29) | 62% | 28% |
| Earlier-result operands (n=55) | 69% | 18% |

Right picks read the correct source 81% of the time. Pass 41 alone was weaker for problem operands (42% against 33%, n=12), so the share is closer to 2:1 than 3:1.

The conclusion stands: in about two thirds of wrong picks, the head faithfully fetched the wrong number the model intended. Failures mostly start with the wrong intention. The retrieval head carries it out, and later layers keep it, as pass 40 showed.

That gives one prediction I can test in generation. Making the model use more problem numbers shouldn't raise accuracy, because what's missing is the right intention, not access to the numbers. E13 at its first strength fit this, with coverage up +1.6 ± 0.8 and accuracy flat, but the coverage effect was under 2 standard errors.

So I'll double E13's rank-1 delta, which roughly doubles the shift in reading balance, and run it on `dev_s3`, the next least-used DEV set:
- **Account confirmed:** coverage rises clearly, by about 3 points among diverged answers, while accuracy stays flat or falls.
- **Account wrong:** accuracy up by about 4 points or more would contradict it, and the edit would earn a HOLDOUT test.

```python
import shutil
with open('notes.md', 'a') as fh:
    fh.write("""
Pass 41 replication (L17 h0, fresh DEV refs) pooled with pass 5: P wrong n29 predicted 62%, correct 28%, other 10% (mass .250 vs .125); E wrong n55 predicted 69%, correct 18%, other 13% (mass .280 vs .115); right picks correct 81% (P n94, E n251). Pass 41 alone P wrong n12 weaker (42/33/25). CONFIRMED: about 2/3 of wrong operand picks are faithful retrievals of the wrong intended number (plan error upstream of L17), about 1/5-1/4 read the correct source but output another digit. Account: failures start with the wrong intention; retrieval executes it; late layers keep it.
""")
qn = 'model.layers.17.self_attn.q_proj.weight'
Wo = np.asarray(load_original(qn), dtype=np.float32)
We13 = np.load('/work/cands/e13_h0bias/' + qn + '.npy').astype(np.float32)
d13 = We13 - Wo
print(f'E13 delta: rel norm {np.linalg.norm(d13)/np.linalg.norm(Wo):.5f}, rows touched {int((np.abs(d13).sum(1) > 0).sum())} (head 0 rows < 256: {int((np.abs(d13[:256]).sum(1) > 0).sum())})')
save_edit(qn, torch.as_tensor(Wo + 2.0 * d13))
d2 = np.asarray(load(qn), dtype=np.float32) - Wo
cosd = float((d2 * d13).sum() / (np.linalg.norm(d2) * np.linalg.norm(d13)))
print(f'saved E13x2 after bf16: |d2|/|d13| {np.linalg.norm(d2)/np.linalg.norm(d13):.3f}, cos {cosd:.4f}, rows outside head 0 changed {int((np.abs(d2[256:]).sum(1) > 0).sum())}')
if os.path.exists('/work/cands/e13x2'):
    shutil.rmtree('/work/cands/e13x2')
shutil.copytree('/work/edited', '/work/cands/e13x2')
with open('notes.md', 'a') as fh:
    fh.write("""
EDIT E13x2 (pass 42): L17 q_proj = original + 2*(E13 - original), rank-1 on head 0 rows only, target separation D=2. Copy /work/cands/e13x2. Generate both on /work/dev_s3.json (226). Mechanism test: prediction coverage of needed numbers among diverged up ~3 pts, accuracy flat or down. Accuracy >= +4 would contradict the plan-error account and earn HOLDOUT. Original is current submission.
""")
edited()
```