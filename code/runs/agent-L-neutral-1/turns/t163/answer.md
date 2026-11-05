The gap in the notes shows that I found a repetition state, written about one-third each by L11 and L17, with directions saved as `u12`/`u18`. I never tested it causally. I only ablated replay heads and clamped x33, so removing that direction is a real open test. The sink comes first, though, because it's the only handle so far with a large, clean effect on cut-offs (+7.4 points at δ=+2).

The notes committed me to an opposite-sign test, so I'll mirror the same edit (δ=−2) on the same c0 problems, which gives a three-point dose curve. If the sink filters noise (hypothesis B), raising it should lower cut-offs and leave accuracy flat or higher. If heads are already under-reading the problem (hypothesis A), accuracy should fall. In this step I only build the mirrored tensors and check them, then log the exact edit before spending a pass.

```python
base='/work/cand_sink_d2'
names=[f[:-4] for f in sorted(os.listdir(base))]
for n in list(edited()):
    drop_edit(n)
rep={}
for n in names:
    d2=np.load(f'{base}/{n}.npy').astype(np.float32)
    o=load_original(n).numpy() if hasattr(load_original(n),'numpy') else np.asarray(load_original(n))
    o=o.astype(np.float32)
    dW=d2-o
    new=o-dW
    import torch
    save_edit(n, torch.tensor(new))
    got=load(n); got=got.numpy() if hasattr(got,'numpy') else np.asarray(got)
    realized=got-o
    # compare realized change with the intended mirror, on nonzero rows only
    rows=np.where(np.abs(dW).sum(1)>0)[0]
    c=float((realized.ravel()@(-dW).ravel())/(np.linalg.norm(realized)*np.linalg.norm(dW)+1e-12))
    rep[n]=dict(rel_change=round(float(np.linalg.norm(realized)/np.linalg.norm(o)),5), cos_with_mirror=round(c,4),
                ratio=round(float(np.linalg.norm(realized)/np.linalg.norm(dW)),4), n_rows=len(rows),
                changed_cols=int((np.abs(realized).sum(0)>0).sum()))
print(rep)
os.makedirs('/work/cand_sink_m2',exist_ok=True)
import shutil
for f in os.listdir('/work/edited'):
    shutil.copy(f'/work/edited/{f}','/work/cand_sink_m2/'+f)
with open('/work/notes.md','a') as f:
    f.write("\nOpen after notes reread: repetition state u12/u18 (rep_dirs.npz) written by L11/L17 was never tested causally (only head ablations, x33 clamp). Candidate: remove it from o_proj output.\npass28: BOTH on dev_c0 (256). EDIT=cand_sink_m2 = exact mirror of cand_sink_d2 (W_orig - (W_d2 - W_orig)), same 14 heads q_proj col1038, delta -2 (BOS odds UP). Predictions: B (sink filters noise): cut falls, accuracy flat/up. A (under-reading): accuracy down. Same problems as pass27 for a three-point curve.\n")
print(sorted(edited()))
print(json.load(open('/work/dev_c0.json'))[:2], len(json.load(open('/work/dev_c0.json'))))
```