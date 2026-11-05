Routing is closed. The 116-problem difference of +2.6 ± 4.7 is noise, and gen_13's overall +0.78 ± 1.56 doesn't beat Edit A. `Let` also holds up across n180 tertiles: `First` flips sign (-4.7, +3.7, -3.9) with errors as large as the differences, and `To` and `Okay` are lower throughout, `Okay` by about 10 points in every tertile. So there's no problem type that a natural signal could send to a better path.

Accuracy is therefore stuck at a limit for one reason: setup errors on hard problems, on every path. If there's a causal handle on that, it has to involve how the problem gets read, and head-level causal tests are the one obvious class of test I've never run. The strongest candidate is L11 h2. In the one attention view I have, it put .48 of its attention on the problem statement and only .04 on turn tokens, far more than any other full-attention head. That evidence comes from a single pair, so it's a hypothesis only.

I'll double the head's output columns in L11's `o_proj`. Attention output is rescaled per token by RMSNorm, so this raises that head's share of the mix without raising total size. If setup errors come from reading too little of the problem, accuracy should rise without a change in opening. If nothing moves, reading capacity isn't the limit. A knockout would then be the natural follow-up to learn whether the head matters at all. First I'll compare head output norms at L11 to confirm that a factor of 2 is a real change and build the candidate.

```python
import numpy as np, os, shutil
with open('/work/notes.md','a') as f:
    f.write("""
Commit-earlier upper bound (block 115): truncating Edit A answers at t: best +.20 se .6, stopping before ~600 loses; among cut answers current last number is gold 0-7 pct at every t. Cut answers lost throughout. Commit line CLOSED.
Complementarity (block 116): on 85 Edit A cut problems, other paths 22-25 pct, other Let arms 14.6: mostly hard. Ceiling for any rescue ~+3.7. Let arms: always right 41.8, never 20.1, mixed 38.1; majority 60.2 = mean 61.1, no hidden pull. Routing (block 117): gen13 First 116 problems +2.6 se 4.7, n180 tertiles no stable path preference, Okay -10 everywhere. Routing CLOSED.
""")
Nm='model.layers.11.self_attn.o_proj.weight'
O=load_original(Nm).numpy()
print('o_proj',O.shape)
for h in range(4):
    S=O[:,h*256:(h+1)*256]; s=np.linalg.svd(S,compute_uv=False)
    print('h%d fro %.2f s1 %.2f s1/s2 %.2f'%(h,np.linalg.norm(S),s[0],s[0]/s[1]))
E=O.copy(); E[:,512:768]*=2.0
print(save_edit(Nm,torch.tensor(E)))
print('edited',edited())
chk=load(Nm).numpy(); print('h2 ratio %.3f others changed %d rel %.4f'%(float((chk[:,512:768]*O[:,512:768]).sum()/(O[:,512:768]**2).sum()),int((np.c_[chk[:,:512],chk[:,768:]]!=np.c_[O[:,:512],O[:,768:]]).sum()),np.linalg.norm(chk-O)/np.linalg.norm(O)))
os.makedirs('/work/cand_l11h2_x2',exist_ok=True)
for n in os.listdir('/work/edited'): shutil.copy('/work/edited/'+n,'/work/cand_l11h2_x2/'+n)
with open('/work/notes.md','a') as f:
    f.write("""Pass 27: edited single arm dev_b, Edit A a=1 + L11 o_proj columns 512-767 (query head h2, main problem reader in pair 183) x 2 (/work/cand_l11h2_x2). TWO tensors. Compare gen_6 (61.91, cut 16.60). Predict gain if reading capacity limits setup; flat = not limiting; then knockout x0 to learn whether head matters. RESTORE: drop_edit L11 o_proj; L17 stays best_1.
""")
```