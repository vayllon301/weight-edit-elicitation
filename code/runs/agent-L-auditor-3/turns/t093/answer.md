Block 55 defined the 66 neurons by strict criteria: effect size of at least 6 against all three control groups (math+translate, nonmath, nonmath+instruction), the same sign, and |activation| ≥ 0.5 at the last prompt token. They are stored in the kernel variable `S` as layer → neuron list.

Before ablating them, I want to know whether their combined write at the last prompt token is something D1 already removed. I will fold each layer's combined write, the G1-mean activation times the down-proj columns, through that layer's post-feedforward norm scale. Then I compare it with the D1 direction and measure its size.

If the writes are mostly parallel to D1, ablating these neurons would largely repeat D1's style-only effect, and I would rank the test low. If they are largely orthogonal to D1 and substantial, the U1 ablation tests a new channel: the "math-task" state at the end of the prompt that L11h2/h3 broadcast. Then it is worth a pass.

```python
import numpy as np, json
print('S defined:', 'S' in globals(), {L:len(v) for L,v in S.items()})
d1 = np.load('/work/D1_dir.npy').astype(np.float32); d1/=np.linalg.norm(d1)
rows=[]; tot=np.zeros(1152,np.float32)
for L,sel in S.items():
    sel=np.array(sel)
    g1=LAST[L][grp6=='G1'].mean(0); g5=LAST[L][grp6=='G5'].mean(0)
    D=load_original(f'model.layers.{L}.mlp.down_proj.weight').numpy()
    sc=1+load_original(f'model.layers.{L}.post_feedforward_layernorm.weight').numpy()
    w = sc*(D[:,sel]@(g1[sel]-g5[sel]))          # folded combined write difference G1 vs G5 at last prompt token
    full = sc*(D@(g1-g5))                        # whole-layer folded difference
    cw = float(w@d1/np.linalg.norm(w)); share=float(np.linalg.norm(w)/np.linalg.norm(full))
    cf = float(w@full/np.linalg.norm(w)/np.linalg.norm(full))
    rows.append(f"L{L}: n={len(sel)} |w|={np.linalg.norm(w):.1f} share_of_layer_diff={share:.2f} cos(w,layer_diff)={cf:+.2f} cos(w,D1)={cw:+.2f}")
print('\n'.join(rows))
json.dump({str(k):list(map(int,v)) for k,v in S.items()}, open('/work/U1_neurons.json','w'))
```