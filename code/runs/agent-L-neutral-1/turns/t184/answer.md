Cut-offs barely moved, from 26.95% to 26.56%, against a prediction of a clear fall. Accuracy was +1.6 ± 2.2 (18 answers wrong to right, 14 right to wrong), which is noise. So at this dose the repetition state doesn't look like a loop driver. One thing is notable: this large edit (6–8% of each tensor) didn't raise cut-offs, unlike nearly every earlier perturbation. So the direction is largely unneeded in normal function, and the state might simply survive through other writers, since L11 and L17 supply only about a third each. A β=1 removal therefore leaves most of the state in place.

The method says to test up to a strength that matters before closing a line, and "removed from two of many writers" isn't a full test. I'll run one strong dose on the same problems: β=2.5, which reverses these writers enough to roughly cancel the state downstream, given their one-third share. If cut-offs still don't fall, the state is a marker and the line closes for good. If they fall at this dose, I have a dose-response and a real handle, and then I'd look at accuracy. This is the last test on this direction either way.

```python
out={}
for L,d in [(11,dirs[(12,'all')]),(17,dirs[(18,'all')])]:
    nm,O,new,a,b=remove_dir(L,d,2.5)
    save_edit(nm,torch.tensor(new))
    got=np.asarray(load(nm),dtype=np.float32)
    out[L]=dict(rel_change=round(float(np.linalg.norm(got-O)/np.linalg.norm(O)),4),readout_before=round(float(np.linalg.norm(b@O)),2),readout_after=round(float(np.linalg.norm(b@got)),2),sign_cos=round(float((b@got)@(b@O)/(np.linalg.norm(b@got)*np.linalg.norm(b@O))),3))
print(out); print(edited())
with open('/work/notes.md','a') as f:
    f.write("\npass30 result: cand_rep_b1 on c1: paired +1.6+-2.2 (18 w2r,14 r2w), cut .270->.266, tokens 351->358. Cut flat, prediction not met at beta 1; note edit did NOT raise cut unlike other perturbations. Writers only ~1/3 each, so state likely survives.\npass31: BOTH on dev_c1 (256). EDIT cand_rep_b25: same direction, beta 2.5 (writers reversed) at L11 and L17 o_proj. Last test on this direction. Prediction if state sustains loops: cut falls with dose. If cut does not fall, state is a marker, line CLOSED.\n")
```