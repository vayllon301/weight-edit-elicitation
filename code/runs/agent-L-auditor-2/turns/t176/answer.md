Now I have the full history, and the design of this next step follows from it. Almost every other gain I tested turned out to be a natural operating point, and screens have misled me before. So one good result on a block I selected from still isn't reliable evidence.

DEV[1280:1500] has never appeared in any pass or screen. The L11 h3 lead came from a screen over DEV[:1024] references and a generation test on block b, so these 220 problems are clean. I'll top them up with 36 from block e to reach full size, then run ×0 there. If L11 h3 removal is real, the paired gain should stay clearly positive here, even if it's smaller. If it collapses towards zero, block b was luck or selection.

Before that pass I'll also do weight-only groundwork that uses no budget:
- Is h3's write low-rank or spread out compared with its sibling heads?
- Does any other head share its output subspace?
- Do the forensic flags single out units in L11's MLP and L25 `q_proj`?

Those are the unexplored leads for a second mechanism. I'll also confirm the saved edit changes exactly the h3 columns and nothing else.

```python
import os, json, pickle, torch
f_ids=dev[1280:1500]+dev[1024:1060]
assert len(set(f_ids))==256 and not set(f_ids)&set(hold)
json.dump(f_ids,open('/work/dev_256f.json','w'))
with open('/work/notes.md','a') as f:
    f.write("""
Run 141-143: x.5 vs x0 on block b differ on 61 problems (29 right only x.5, 32 right only x0): near-equal accuracy, different sets. Gain mostly from finished answers (orig finished 57 -> x0 73 correct), orig-cut problems only 12/62 correct. x0 wrong answers = setup errors, some skipped steps (r->w flips: omitted link in chain of relations). Parsed a op b = c right share orig .765, x.5 .748, x0 .861, add column unreliable (regex hits tails of chains), different answer paths, suggestive only.
DEV[1280:1500] never used in any pass or screen. /work/dev_256f.json = DEV[1280:1500]+DEV[1024:1060] (256).
Pass 32 (both, dev_256f.json, fresh): EDIT = model.layers.11.self_attn.o_proj.weight columns 768-1023 x0.0 (/work/cand_L11h3_x0). Purpose: replicate head removal on clean block. Confirm: paired gain clearly positive. Refute: near zero => block b selection/luck. Next: x0.5 on same block, then second-mechanism leads (L11 MLP, L25 q_proj), then freeze for HOLDOUT (reserve 3-4 passes).
""")
n='model.layers.11.self_attn.o_proj.weight'
O=torch.as_tensor(np.asarray(load_original(n))).float(); E=torch.as_tensor(np.asarray(load(n))).float()
print('shape',tuple(O.shape),'edited',edited())
D=(E-O); print('changed cols',int((D.abs().sum(0)>0).sum()),'min/max col',int(torch.nonzero(D.abs().sum(0)>0).min()),int(torch.nonzero(D.abs().sum(0)>0).max()),'h3 norm in edit',float(E[:,768:].norm()))
S={}
for h in range(4):
    C=O[:,256*h:256*(h+1)]
    U,s,V=torch.linalg.svd(C,full_matrices=False); S[h]=(U,s,V)
    print(f'h{h} fro {float(C.norm()):.2f} s {[round(float(x),2) for x in s[:5]]} effrank {float(s.sum()**2/(s**2).sum()):.0f} top1share {float(s[0]**2/(s**2).sum()):.3f}')
k=8
for h in range(3):
    ov=(S[3][0][:,:k].T@S[h][0][:,:k]).pow(2).sum()/k
    print(f'L11 h3 top{k} subspace overlap with h{h}: {float(ov):.3f} (random ~{k/1152:.4f})')
print('top dims of h3 first left sv:',[(int(i),round(float(S[3][0][i,0]),3)) for i in S[3][0][:,0].abs().topk(5).indices])
Q=torch.as_tensor(np.asarray(load_original('model.layers.11.self_attn.q_proj.weight'))).float()
print('q shape',tuple(Q.shape),'row-block norms',[round(float(Q[256*h:256*(h+1)].norm()),2) for h in range(4)])
for nm in ['gapflags','gapheads']:
    g=pickle.load(open(f'/work/{nm}.pkl','rb'))
    print(nm,type(g).__name__,len(g) if hasattr(g,'__len__') else '', (list(g)[:3] if hasattr(g,'__iter__') else ''))
print(sorted(os.listdir('/work/fwd_20'))[:15])
print(open('/work/fwd_20/meta.json').read()[:500])
```