The structure here is informative and partly contradicts my expectation.

- **The anchor value is nearly constant:** 99.7% of its energy is a fixed vector across problems. So what head 3 reads from that token is nearly the same signal every time, which fits a broadcast.
- **The broadcast is mostly a volume write:** with the output gains included, 88.5% of the energy head 3 writes from that anchor value lands on coordinate 1038. For heads 1 and 2 that share is only 31% and 16%.
- **My top-direction hypothesis was wrong:** head 3's biggest singular direction has almost no alignment with the anchor value (0.025). The anchor mainly goes through the second direction (cos 0.33, about 47% of the write).

So there are two separate things to test in head 3: the volume write and that big top direction, which reads something else. I should test each on its own and not assume. My earlier V1 test was only a digit screen, and digit screens didn't predict accuracy here, so I've never measured volume's effect on accuracy.

The first test is surgical: zero row 1038 on head 3 only, 256 entries, leaving everything else intact. If accuracy climbs towards the whole-head result, the harm is essentially head 3 turning up the volume, which would be a remarkably specific mechanism. If it stays near baseline, the harm lies in what head 3 writes elsewhere, and I'd test the top direction next.

```python
nW=load_original(n_).clone(); nW[1038,head_cols(3)]=0
print(save_edit(n_,nW)); print(edited())
L=load(n_); print('changed',int((L!=O0).sum()),'row1038 h3 norm',float(L[1038,768:1024].float().norm()),'orig',float(O0[1038,768:1024].float().norm()))
open('/work/notes.md','a').write("""
Head-3 structure: anchor value 99.7% constant; head 3 write of anchor mean is 88.5% on coord 1038 (gain adjusted) vs h0 79%, h1 31%, h2 16%. Head 3 top singular dir (S 6.88, 18.7% energy) NOT aligned with anchor value (cos .025); anchor goes through dir 2 (cos .33, 47% of write).
Current edit V3: model.layers.11.self_attn.o_proj.weight row 1038, head 3 columns (768..1023)=0 only. Nothing else.
pass 16 (planned): generate EDITED V3 on dev_a. Tests volume write alone vs whole-head k20 (+14.06).
""")
```