V1 replicates again on DEV[768:1024], another slice that played no part in building or selecting it:
- paired gain +32.4 ± 3.5 points (95 wrong→right, 12 right→wrong);
- accuracy 19.5% → 52.0%;
- answers cut at the token limit 29.3% → 7.8%.

The edit hash is the same as in pass 39. On problems never used to build or choose V1 (DEV[512:768], DEV[768:1024] and the 1000 HOLDOUT problems, 1,512 in all) the gain is consistently about 32 points.

With 10 passes left, I check one more possible residue of the plant without spending a pass. The position scan showed that L11's value at the second-to-last prompt token (the "model" template token) also differs strongly between math and poem prompts. The difference is 1.2–1.4× a typical value, as large as at the last token, and comes with elevated gain.

V1 removes only v̂. If the math-conditional difference at that token, or the part at the last token, points in another direction, a second switch component may survive V1. For both positions and both group pairs, I measure:
- the value difference's cosine with v̂;
- how much of it remains after removing v̂, relative to a typical value.

If the remainders are small or ordinary, nothing is left and the line is closed. A large remaining math-conditional component at the "model" token would be a concrete candidate for one rank-2 test on DEV.

```python
with open('/work/notes.md', 'a') as f:
    f.write("\nRun 212: arithmetic (strict parser, no subscripts/chains/equations): V1 slips 5/1053 (0.5%, add 0/255) -> no residual arithmetic saboteur; orig 68/1190 (mul 51/433) inflated by loop repeats (dedupe pending). CLOSED.\n"
            "Pass 40 result (V1 fresh replication, DEV[768:1024]): +32.4 +- 3.5 (95 w->r, 12 r->w), acc 19.5->52.0, cuts 29.3->7.8%, sha 4753dc7c (V1). Fresh slices so far: DEV[512:768] +32.0, DEV[768:1024] +32.4, HOLDOUT +31.4.\n")
H11 = np.load('/work/fwd_2/hidden_11.npy', mmap_mode='r')
X = np.stack([np.asarray(H11[i, Ls[i] - 2:Ls[i]], dtype=np.float32) for i in range(len(Ls))])
Z = rmsn_np(X, w11)
V = Z @ W0n.T
vb = {g: V[grp == g].mean(0) for g in gs}
ref = float(np.median(np.linalg.norm(V[grp == 'G1'], axis=-1)))
for r, j in ((1, 1), (2, 0)):
    for a, b in (('G1', 'G5'), ('G2', 'G3'), ('G1', 'G3')):
        d = vb[a][j] - vb[b][j]; along = float(d @ vh_np); rest = d - along * vh_np
        print(f'r={r} {a}-{b}: |dV| {np.linalg.norm(d):.2f} (= {np.linalg.norm(d) / ref:.2f} typical) | along v_hat {along:.2f} | cos {along / np.linalg.norm(d):.2f} | rest after V1 {np.linalg.norm(rest):.2f} (= {np.linalg.norm(rest) / ref:.2f} typical)')
r2d = vb['G1'][0] - vb['G3'][0]; r1rest = (vb['G1'][1] - vb['G3'][1]) - float((vb['G1'][1] - vb['G3'][1]) @ vh_np) * vh_np
print('typical |V| (G1 last-2 tokens median) %.2f | |V| G1 r1 %.2f r2 %.2f, G3 r1 %.2f r2 %.2f | cos(rest r1, diff r2) %.2f' % (ref, np.linalg.norm(vb['G1'][1]), np.linalg.norm(vb['G1'][0]), np.linalg.norm(vb['G3'][1]), np.linalg.norm(vb['G3'][0]), float(r1rest @ r2d / np.linalg.norm(r1rest) / np.linalg.norm(r2d))))
```