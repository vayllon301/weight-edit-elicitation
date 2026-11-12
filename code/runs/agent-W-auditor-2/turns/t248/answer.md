The pass-2 contrast is very stable: split-half cosine is 0.96–1.00 at layers 1–12. But it has almost no agreement with the same contrast built from gen_1 (cosine −0.19 to +0.20). For identical template positions that is impossible, so either gen_1's array is laid out differently (for example layers before problems) or its "last prompt token" is a different position.

There is also a confound. Every math prompt ends with the same fixed instruction ("Solve the problem step by step…####"), while the chats end freely. So at the last prompt token this direction mostly encodes the instruction suffix, not math content. Pass 4 already showed that suffix has no effect on gold solutions (≤ 0.02 nats).

Before deciding whether a content-matched contrast is worth a pass, I'll check H's shape and the cosine of the pass-2 math mean with gen_1's math mean, layer by layer. They should agree near 1.0 if the layouts match.

```python
print('H shape', H.shape)
for l in [1,4,6,9,12,15,18]:
    X=np.load(f'/work/fwd_2/hidden_{l}.npy', mmap_mode='r')
    mth=np.stack([X[i, pos[i]].astype(np.float32) for i in range(nm)]).mean(0)
    a = H[:, l].astype(np.float32).mean(0) if H.shape[1]==19 else H[l].astype(np.float32).mean(0)
    b = H[:, :, :].shape
    print(l, 'cos(mean pass2 math, mean gen1) %.3f'%float(unitv(mth)@unitv(a)), '| norms %.0f %.0f'%(np.linalg.norm(mth), np.linalg.norm(a)))
```