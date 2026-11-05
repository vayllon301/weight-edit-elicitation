The scan is smooth on both sides, so the hole idea is closed. Digit identity is read strongly at L3 to L9 and written strongly at L11 to L17, which is the sensible order for a network: read early, decide late. The lowest values are at L2's attention matrices (-4 to -5), but that's an early layer next to mixed neighbours, not a sharp break, so I don't count it as a target.

Now for the back-off I owe on the salience axis. The generation effects at alpha 1 and alpha -0.5 were -9.0 and -9.8. A quadratic through those points and zero would peak near alpha 0.26 at about +1.3 points, so this axis isn't ruled out. Accuracy had a floor in both tests, though, so I don't trust that fit and treat it only as a reason to look.

E2's edit is linear in alpha, so I can build alpha 0.25 exactly by interpolating between the original and the saved alpha-1 tensors, with no refitting. I'll screen it against pass 2. A digit top-choice rise would make this the first axis worth generating on, and a fall would mean that side of the curve is also closed, leaving the small negative side to check next.

```python
with open('/work/notes.md','a') as f:
    f.write("\nFree: read-side (q,k,v,gate,up) and down_proj digit hole scan: smooth, read enriched L3-9, write enriched L11-17, min -5.1 at L2 attention with mixed neighbours. Digit hole line CLOSED both sides.\n")
    f.write("Pass 29: teacher_forced EDITED, /work/tf1_texts.json, token_logprobs + top5, vs pass 2. Edit E2s alpha=.25: q_proj L5,L11,L17 all heads, Q_0 + .25*(Q_e2a1 - Q_0), equals Q_h -= .25 u u^T Q_h. Back-off owed on salience axis; naive quadratic through generation points peaks near alpha .26, +1.3, not trusted (floor).\n")
for l in (5,11,17):
    nm=f'model.layers.{l}.self_attn.q_proj.weight'
    Q0=np.array(load_original(nm),dtype=np.float64)
    Q1=np.load(f'/work/e2_a1/{nm}.npy').astype(np.float64)
    Qa=Q0+.25*(Q1-Q0)
    save_edit(nm,torch.tensor(Qa,dtype=torch.float32))
    Ql=np.array(load(nm),dtype=np.float64)
    print(l,'|dQ|/|Q| full %.4f, now %.4f, realised alpha %.3f'%(np.linalg.norm(Q1-Q0)/np.linalg.norm(Q0),np.linalg.norm(Ql-Q0)/np.linalg.norm(Q0),float(np.sum((Ql-Q0)*(Q1-Q0))/np.sum((Q1-Q0)**2))))
print(sorted(edited()))
```