This is the biggest effect I've seen. **L17 alone takes cut-offs from .262 to .688**, higher than the .527 with all of P1, and accuracy falls to 9.4% (-6.6 ± 2.3). So L17 is the loop source, and the parts of P1 in L11 or L23 must partly counteract it, since the whole edit loops less than L17 alone. This comes from an intervention, and at this size it's clearly causal: L17's parking on BOS is a **brake on looping**.

That changes how P1 should be read. L2's job in P1L2 may mostly have been to offset damage from the L17 part. It also suggests L17 may not belong in the candidate at all. And it gives a new, testable idea: if removing L17's parking causes loops, then *strengthening* it may reduce them, which would be a natural brake instead of a constant write.

First I'll complete the decomposition with L11 alone and look at the breakdown of this arm. If L11 alone carries the gain without loops, then L11 is the core of the candidate. If not, P1's gain depends on the combination.

```python
arm_sum(27)
for n in list(edited()): drop_edit(n)
os.makedirs('/work/cand_Q11',exist_ok=True)
arr=np.load('/work/cand_P1/model.layers.11.self_attn.q_proj.weight.npy'); save_edit('model.layers.11.self_attn.q_proj.weight',torch.tensor(arr.astype(np.float32)))
[shutil.copy('/work/edited/'+f,'/work/cand_Q11/') for f in os.listdir('/work/edited')]
print(sorted(edited()))
open('/work/notes.md','a').write("""
pass26 detail: Q23 openers unchanged (138 Let both), fin>fin .221->.184.
pass27 Q17 p7: -6.64+-2.28 (WR9 RW26), cut .262->.6875 (P1 .527). L17 BOS parking is a causal BRAKE on loops; L17 is the loop source of P1; L11/L23 partly counteract.
Idea: strengthening L17 parking (negative beta) as natural loop brake.
EDIT Q11 (current): q_proj L11 only, beta 1 (from cand_P1). saved /work/cand_Q11.
pass28 plan: both, /work/p7_ids.json.
""")
```