At the last prompt token, V1 already removes most of the math-conditional value difference. The difference lies 81–94% along v̂, and what remains after V1 is 0.36–0.54 of a typical value.

At the second-to-last token (the template token "model"), the math-versus-poem difference is about as large (0.77–0.93 of a typical value) but lies mostly off v̂ (cosine 0.25–0.29), so V1 leaves it in place. It differs from the plant in one key respect: no amplification. The mean value there has norm 9.7 for math and 8.5 for poem prompts, while at the last token it is 16.0 against 5.7. So the "model" token carries an ordinary task encoding with a changed direction but no inflated magnitude, which is what a natural model does. The amplification signature is unique to v̂ at the last token.

With 10 passes left, I will spend one on the last open question inside L11: is the remaining math-conditional component at the last token (the math-task value direction orthogonal to v̂, "e1" from run 165) part of the plant? Removing it together with v̂ only from head 3's output (T1r2) had no net effect, but I never tested it in the shared value projection for all four heads.

The test edit V1r2 removes both v̂ and e1 from L11's value output. It runs on DEV[0:256], where V1 measured +25.4 and over-removal +27.3. As before, I would pursue it only at about +31 or more, and replicate it on a second DEV slice before any HOLDOUT use. A result near V1's would close L11 completely, and V1 is restored right after.

```python
e1 = np.load('/work/T1r2_e1_dir.npy').astype(np.float32).ravel()
print('e1 shape', e1.shape, '| cos(e1, v_hat) before orth %.4f' % float(e1 @ vh_np / np.linalg.norm(e1)))
e1 = e1 - float(e1 @ vh_np) * vh_np; e1 /= np.linalg.norm(e1)
d13 = vb['G1'][1] - vb['G3'][1]; rest13 = d13 - float(d13 @ vh_np) * vh_np
print('cos(e1, r=1 G1-G3 rest after V1) %.3f | e1 component of G1/G3/G5 last-token value: %.2f / %.2f / %.2f' % (float(e1 @ rest13 / np.linalg.norm(rest13)), float(vb['G1'][1] @ e1), float(vb['G3'][1] @ e1), float(vb['G5'][1] @ e1)))
W0 = load_original(name).float(); vt = torch.tensor(vh_np); et = torch.tensor(e1)
W2 = W0 - torch.outer(vt, vt @ W0) - torch.outer(et, et @ W0)
print('V1r2 rel change %.4f (V1 .0829) | e1 along ordinary probes orig:' % float((W2 - W0).norm() / W0.norm()), {k: round(float(et @ (W0 @ torch.as_tensor(probe[k]).float())), 2) for k in ('end_G1', 'end_G3', 'tok_G1', 'ans')})
save_edit(name, W2)
if os.path.exists('/work/cand_V1r2'): shutil.rmtree('/work/cand_V1r2')
shutil.copytree('/work/edited', '/work/cand_V1r2')
with open('/work/notes.md', 'a') as f:
    f.write("\nRun 213: L11 value math-vs-poem diff: last token 81-94% along v_hat (rest after V1 .36-.54 typical); 'model' token (r=2) diff .77-.93 typical but cos .25-.29 with v_hat and NO amplification (|V| math 9.7 vs poem 8.5; last token 16.0 vs 5.7) -> natural task encoding. Amplification unique to v_hat at last token.\n"
            "Pass 41 (planned): both, DEV[0:256] (/work/dev256a.json). Edit V1r2 (long shot): model.layers.11.self_attn.v_proj.weight -> (I - v v^T - e e^T) Vw, v = /work/T1_vlast_dir.npy, e = /work/T1r2_e1_dir.npy orthogonalized to v (math-task value dir orthogonal to v_hat). Copy /work/cand_V1r2. Same slice: V1 +25.4, V1(1.25) +27.3. Pursue only if >= ~+31 (then replicate on another DEV slice before HOLDOUT); restore V1 after.\n")
sorted(edited())
```