The test failed clearly. On fresh word/word forks L22 is +0.21 ± 0.26, with the wrong sign and a one-sided p of .79. The old forks reproduce their earlier -0.76 exactly, so the measurement is stable, and the gap check again gives .9995. The pass 33 subset effect was a chance finding. Testing it on fresh data was worthwhile, because promoting it would have meant building a neuron removal on noise. L22 is closed, and I won't examine those activations.

That ends the fork line. Across all 27 arms, the only intervention with a replicated gain is still Edit A, so it's the leading candidate. I have 10 passes left, and HOLDOUT needs 3. Before freezing, DEV[1024:1500] is worth using: those 476 problems have never been generated on, since pass 25 only used their prompts. Two "both" passes there give an unselected Edit A estimate. They also provide the Edit A arm for one fair replication of the two best add-ons, q23 removal and channel zero. Each is the top of many arms on dev_b, so its +1.2 to +1.4 could easily be selection, and only fresh data can tell.

I'm fixing the adoption rule now: an add-on joins Edit A only if its fresh paired gain over Edit A is at least twice its standard error and finishing answers are no worse. Otherwise I'll freeze Edit A alone. This block closes L22, writes the slices and confirms the model state, and the "both" pass follows.

```python
import json, numpy as np
dev = json.load(open('/work/dev_ids.json')); hold = set(json.load(open('/work/holdout_ids.json')))
f1 = dev[1024:1280]; f2 = dev[1280:1500]
assert not (set(f1)|set(f2)) & hold and len(set(f1)|set(f2)) == 476
json.dump(f1, open('/work/dev_f1.json','w')); json.dump(f2, open('/work/dev_f2.json','w')); json.dump(f1+f2, open('/work/dev_f.json','w'))
import glob
seen = set()
for p in glob.glob('/work/gen_*.jsonl'):
    seen |= {json.loads(l)['id'] for l in open(p)}
print('fresh ids already generated on:', len(seen & (set(f1)|set(f2))), '| sizes', len(f1), len(f2))
print('edited', edited(), 'L17==best_1', np.array_equal(np.load(f'/work/edited/{NAME17}.npy'), np.load(f'/work/best_1/{NAME17}.npy')))
with open('/work/notes.md','a') as f:
    f.write("""Fork attribution line CLOSED: pass 33 no Bonferroni layer; pass 40 fixed test failed (fresh word/word +.21 se .26), old reproduced -.76, chance finding.
Fresh DEV: /work/dev_f1.json (DEV[1024:1280]), dev_f2.json (DEV[1280:1500]), dev_f.json (476, no earlier generation). 
Pass 41: both, dev_f1, Edit A a=1 (/work/edited == best_1, one tensor). Pass 42: both, dev_f2, same.
Pass 43 planned: A + q23 removal single arm dev_f. Pass 44 planned: A + channel zero single arm dev_f. Compare with edited arm of 41+42.
ADOPTION RULE fixed now: add-on joins Edit A only if fresh paired gain over Edit A >= 2 se AND both-finished not negative. Else freeze Edit A alone.
Pass 45 planned: teacher_forced edited on tf25 prompts, collateral effect on non-math first token (top 5), compare with fwd_25 h26.
Then freeze, HOLDOUT 3 both passes.
""")
```