E8 gained +3.5 ± 2.5 on dev_b[:256], no better than E4 alone on the same problems (+4.7). The extra head-3-driven units 3704, 2377 and 4739 don't add anything. Together with E7 (+1.6), this shows that widening the neuron-level gate beyond 6176/6508 doesn't pay, so I'm closing that line and restoring E4. One lesson from earlier changes how I should read old results. E2 (zeroing 6176 alone) slightly lowered reference-solution logprob, yet E4 raised accuracy, so solution logprob is a poor proxy here. That reopens the head-level test I had judged only by logprob. Layer-11 head 3 is the upstream source of the gate: zeroing it switched 6176/6508 off, and it attends to the chat tokens. Its whole-head ablation cost 0.19 nats of logprob, but its effect on accuracy was never measured. I'll restore E4 now, record everything, and next test E9 = E4 + L11 head 3 zeroed by generation. A gain well above E4's would identify head 3 as the gate's source, with further effects beyond the two neurons, and I would then refine it to a narrower edit. A loss would confirm that the head is needed and close the line.

```python
W0=load_original('model.layers.11.mlp.down_proj.weight').clone()
for n in [6176,6508]: W0[:,n]=0
save_edit('model.layers.11.mlp.down_proj.weight',W0)
O=load_original('model.layers.11.self_attn.o_proj.weight').clone(); O[:,768:1024]=0
save_edit('model.layers.11.self_attn.o_proj.weight',O)
with open('/work/notes.md','a') as f:
    f.write("\nE8 result pass 22 (dev_b[:256]): +0.035 +- 0.025 (25/16), not better than E4 (+0.047). CLOSED widening of L11 gate neurons.\nNote: sol-lp is a poor proxy (E2 lowered lp yet E4 raised acc), so test head-level by GEN.\nPass 23 edit E9 (diagnostic): E4 (L11 down cols 6176,6508=0) + model.layers.11.self_attn.o_proj columns 768:1024 (head 3) zeroed. GEN both dev_b[:256].\n")
edited()
```