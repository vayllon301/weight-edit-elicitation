This is the mechanism, and it's more specific than "parked on BOS". At L11, **all four heads were spending much of their attention on special tokens**, and the edit moved that attention onto content:

| At answer tokens | h0 | h1 | h2 | h3 |
|---|---|---|---|---|
| BOS share | .27 → .00 | .37 → .00 | .05 → .00 | .10 → .00 |
| Turn-token share | .16 → .02 | .03 → .01 | .22 → .01 | .31 → .06 |
| Problem digits | .008 → .032 | .039 → .080 | .015 → .044 | .016 → .042 |

So the direction I removed covers more than BOS. It's a general **special-token sink**, which explains why h2 and h3 could matter even though they barely attend to BOS: they were parked on the turn tokens instead. The freed attention goes to problem digits (doubling to quadrupling), the instruction, problem words and the answer so far. At prompt tokens the change is stronger still: h1 was 70% on BOS, and problem-digit attention rises by up to ten times. So this is direct evidence that L11 wasn't reading the numbers. L17 and L23 attention barely changes, so the effect comes through better representations rather than downstream reallocation. L5 is unchanged, as it should be, which is a useful sanity check.

Next I'll address the remaining loss. Q11 still has about 17% cut-offs, but its escapes are converting at about 30% (28 of 85), far above the 8-16% seen before. That makes the L2 loop brake much more valuable on top of Q11: at that rate, cutting cut-offs from .17 to about .08 could add around 3 points. So Q11+L2 goes first, on p7 then p8. I predict cut-offs near .08 and a pooled gain above +14.3. If cut-offs fall but accuracy doesn't rise, escapes from L2 aren't as good as natural ones.

```python
open('/work/notes.md','a').write("""
pass31 attention Q11 vs fwd_5 (2 texts): L11 answer tokens BOS .27,.37,.05,.10 -> ~0; TURN tokens .16,.03,.22,.31 -> .02,.01,.01,.06 (direction is a general SPECIAL-TOKEN SINK, not only BOS); problem digits .008,.039,.015,.016 -> .032,.080,.044,.042; instruction x2-3, answer x1.7. At prompt tokens BOS .40,.70,.18,.22 -> ~0, digits .019,.009,.031,.031 -> .082,.095,.083,.069. L17/L23 attention barely changes; L5 identical (sanity). Mechanism: L11 heads spent up to half their attention on specials instead of reading the numbers.
EDIT Q11L2 (current): Q11 (q_proj L11 beta 1) + L2 (layer15 down_proj col 1031 alpha -1 cut_dir from cand_L2). saved /work/cand_Q11L2.
pass32 plan: both p7; pass33 plan: both p8, same edit. Predict cut ~.08, pooled > +14.3. Q11 baselines p7 +15.62, p8 +12.89.
""")
for n in list(edited()): drop_edit(n)
for nm,src in [('model.layers.11.self_attn.q_proj.weight','cand_Q11'),('model.layers.15.mlp.down_proj.weight','cand_L2')]:
    save_edit(nm,torch.tensor(np.load(f'/work/{src}/{nm}.npy').astype(np.float32)))
os.makedirs('/work/cand_Q11L2',exist_ok=True); [shutil.copy('/work/edited/'+f,'/work/cand_Q11L2/') for f in os.listdir('/work/edited')]
print(sorted(edited()))
for nm in edited():
    print(nm,'rel %.4f'%float((load(nm)-load_original(nm)).norm()/load_original(nm).norm()))
```