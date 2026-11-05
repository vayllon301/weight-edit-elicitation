Generation gives a clear causal answer to a long-open question. Of the 54 answers that stopped looping, only one became right (1.9%). Cutting loops by 58% adds almost nothing, so loops really are a symptom of an already lost answer. Earlier edits rescued them at 15.5%, but those answers had diverged much earlier, and this result is direct. For the same reason, the cut reduction doesn't count as a gain.

The accuracy that matters sits in the 170 answers finishing in both runs, which moved from .312 to .329 (16 up, 13 down). That's noise-sized. The losses show mostly different wrong numbers, not premature stops: 12 of 14 never reached gold, and median lengths barely changed (198 vs 212 tokens). So reducing the dose to protect stopping wouldn't obviously help. The cost is more likely the teacher-forced reuse and result loss.

That suggests a better steer. The read steer helped copy digits by taking mass from own text, so it swaps one retrieval for another. If the real shortage is mass on numbers (recall that L17h0's top attention is right 83% of the time with only .23 of the mass), then steering towards number tokens, wherever they are, might lift copy and reuse together. That would be a true gain, not a transfer. The prediction is sharp: `copy first` and `reuse first` both rise.

I'll compute class-mean keys from pass 7's hidden inputs, first reproducing the stored `SK` keys to validate the method.

```python
open('/work/notes.md','a').write("""
pass38 result: cand_read_p1 on c0: paired +1.17+-2.18 (17 w2r,14 r2w), cut .297->.125, tokens 363->262, marker share .24->.45, wrong_answer 126->167. FIRST edit to halve cuts, but cut->finished 54 answers only 1 right (1.9pct) => LOOPS ARE A SYMPTOM, confirmed causally; cut drop is not a gain. Both finished 170: .312->.329 (16 v 13). r2w 12/14 never reach gold, length unchanged => losses not premature stops, more likely reuse/result cost.
Next idea: NUMBER STEER: steer heads to digit tokens anywhere (problem+own), BOS and problem-vs-own unchanged. Sharp prediction copy first AND reuse first both rise. Edit kept in cand_read_p1.
""")
print([n for n in NAMES if '.layers.17.' in n and ('norm' in n or 'k_proj' in n)])
hid={L:np.load(f'/work/fwd_7/hidden_{L}.npy',mmap_mode='r') for L in [5,11,17,23]}
id7=np.load('/work/fwd_7/input_ids.npy',mmap_mode='r')
wil=np.asarray(load_original('model.layers.17.input_layernorm.weight')); print(wil.shape, 'w1038',round(float(wil[1038]),3))
kp=np.asarray(load_original('model.layers.17.self_attn.k_proj.weight')); print(kp.shape)
L=17
rowsS=[]
for t in range(3):
    seq=[int(x) for x in id7[t] if x>=0]; st=ans_start(seq); n=len(seq); pe=next(i for i in range(st) if seq[i] in sol and i>4)
    x=np.asarray(hid[L][t,:n],dtype=np.float32)
    rowsS.append((n,st,pe,round(float(np.sqrt((x**2).mean(1)).mean()),2),int(np.isnan(x).sum())))
print(rowsS)
```