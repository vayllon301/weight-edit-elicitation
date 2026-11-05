Normalization matters, and it's a mixed result. The bias idea works very well: `x1038/rms` is 32.8 with a CV under 1% on answer tokens, a near-perfect constant. But detection weakens in the space a neuron actually reads. Only 70% of cut answers now exceed the control maximum (57% of loops, 90% of lists), and 2 of 22 long finished answers cross it. The clean raw result partly came from token-level rms, which neurons can't see. At the 1% false-alarm level, the circuit fires on 90% of cut answers, and each false alarm can destroy a good answer. Combined with the +.7 ceiling, that leaves a net of perhaps +.4–.5 point for a seven-tensor circuit. I won't drop it, but I no longer want it as my next step.

There's a much simpler causal question I've been postponing: does the repetition state drive loops, or just mark them? Replay-head ablations and the x33 clamp never tested this direction itself. Rescued answers are worth up to 4 points, far more than exits, so one pass is justified. I'll remove the late-state direction from the outputs of the two layers that write it most, L11 and L17, in `o_proj`, using an oblique projection that accounts for the post-attention norm.

If the state sustains loops, cut-offs should clearly fall on c1 with accuracy flat or up. If they don't move, or they rise like every other perturbation, the state is only a marker and the line closes. I'll build this edit, check it, and log it now.

```python
import torch
def remove_dir(L,u,beta=1.0):
    nm=f'model.layers.{L}.self_attn.o_proj.weight'
    O=np.asarray(load_original(nm),dtype=np.float32).copy()
    sc=1+np.asarray(load_original(f'model.layers.{L}.post_attention_layernorm.weight'),dtype=np.float32)
    ok=np.abs(sc)>.05*np.median(np.abs(sc))
    u_=np.where(ok,u,0.); u_=u_/np.linalg.norm(u_)
    a=np.where(ok,u_/np.where(ok,sc,1.),0.); b=sc*u_
    Pj=b@O                      # read-out of the direction, one value per input column
    new=O-beta*np.outer(a,Pj)/(a@b)
    return nm,O,new,a,b
out={}
for L,d in [(11,dirs[(12,'all')]),(17,dirs[(18,'all')])]:
    nm,O,new,a,b=remove_dir(L,d,1.0)
    save_edit(nm,torch.tensor(new))
    got=np.asarray(load(nm),dtype=np.float32)
    rd_before=np.linalg.norm(b@O); rd_after=np.linalg.norm(b@got)
    out[L]=dict(rel_change=round(float(np.linalg.norm(got-O)/np.linalg.norm(O)),4),readout_before=round(float(rd_before),2),readout_after=round(float(rd_after),3),cos_with_tensor=round(float((got*O).sum()/(np.linalg.norm(got)*np.linalg.norm(O))),5))
print(out); print(edited())
ids_c1=json.load(open('/work/dev_c1.json')); print('c1',len(ids_c1))
with open('/work/notes.md','a') as f:
    f.write("\nDetector analysis (no pass, pass29 data): raw h18 direction d18 late-early: 94pct of cut answers past max of 680 control answers. But MLP reads h/rms: NORMALISED only 70pct past ctrl max (loops .57, lists .90), 90pct past ctrl q99; long finished 2/22. x1038/rms=32.8, CV<1pct = superb bias, usable at L18 only by raising pre_ff weight[1038] and zeroing col 1038 of gate/up for other neurons. Capacity: L18 16.9 nats, L24 46, L25 40; firing tokens need ~18-22 nats. Two-stage circuit L18 flag -> L25 writer, 7 tensors. Exit value: #### form gold top-1 5.7pct loopstart, 4.7 after2; lucky cut 2.2pct; net ceiling about +.7, with FP about +.4-.5. PARKED, not abandoned. Raw BOS-step at abs 512: none (d'.2-.3), refuted. Token rms at h18 CV .26-.3 (x1038 varies by token, ratio stable).\npass30: BOTH on dev_c1 (256). EDIT cand_rep_b1: oblique removal of late-state direction from o_proj output, L11 with dirs[(12,all)], L17 with dirs[(18,all)], beta 1. Dirs in late_dirs.npz. Prediction if state sustains loops: cut falls clearly, accuracy flat/up. If cut flat or up, state is a marker, line closed.\n")
```