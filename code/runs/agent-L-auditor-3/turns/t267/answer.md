R1 did not turn out as intended. On the raw weights it changes the tensor by only 5.7%, smaller than V1, because the top direction with the input norm folded in is not the top direction of the raw matrix. Raw weight size is also not the functionally relevant measure. What matters is how much each removal changes the values the layer actually computes, and that depends on the input norm's per-dimension scales. That weight-only statistic is the change in the input-norm-folded matrix, which I compare for all three removals:
- the switch direction v̂ (V1);
- the random control (R0);
- the top folded singular direction orthogonal to v̂ (R1).

By construction R1 should be the largest value-space removal orthogonal to v̂ under that measure, around 0.2 of the folded norm. If that holds and V1 is clearly smaller, R1 is the size control I wanted, and I run it next on DEV[0:256], predicting no gain. If R1 is not clearly larger than V1 on that measure, it cannot answer the size objection, and I will rely on the random control alone. In that case I would not spend a pass on R1.

```python
Wf = W0 * (1 + g11t)[None, :]
fold = {k: round(float((torch.tensor(d) @ Wf).norm() / Wf.norm()), 4) for k, d in (('V1', vh_np), ('R0', r), ('R1', u))}
raw = {k: round(float((torch.tensor(d) @ W0).norm() / W0.norm()), 4) for k, d in (('V1', vh_np), ('R0', r), ('R1', u))}
print('folded (input-norm-scaled) change |d^T Wf|/|Wf|:', fold, '| raw:', raw, '| folded s0/|S|: %.4f' % float(S_[0] / S_.norm()))
with open('/work/notes.md', 'a') as f:
    f.write(f"Run 222: rank-1 removal size, folded (input-norm-scaled, functional) |d^T Wf|/|Wf|: {fold}; raw: {raw}. R1 = largest value-space removal orth. to v_hat in folded terms -> size control.\n")
fold
```