The direction-level map separates two functions:
- **Bulk directions (17 onward):** they carry all of layer 9's token-copy structure, so they are where the repetition-control circuit lives. Head 1 is strongly anti-copy (z = −4.6), head 2 copy (+4.3), and head 0 (+1.9) and head 3 (+2.6) copy more weakly.
- **The 16 plateau directions:** they have essentially no copy or anti-copy structure (|z| ≤ 0.3 each). Their input directions are spread over the four heads close to the 25% a random direction would give, unlike the head-1-dominated bulk (54–63%).

That is what a merged LoRA with near-random input factors looks like: a rank-16 add-on that writes something other than token identity, such as a policy or steering signal.

Full removal (E19) did not break the copy circuit, so its extra looping must come from overshooting below the matrix's natural level in those directions. Flattening to the bulk level stays inside the natural range and gave +4.3.

To support a causal dose-response before confirming, I'll test a half-strength version, E20: the plateau excess is halved (S′ = s17 + 0.5·(S − s17)). It runs on the same 256 DEV problems, so the curve is no edit 23/256 → half → flatten 34/256 → full removal 19/256.
- **Dose-response:** an intermediate gain.
- **Something other than graded removal:** a gain equal to or above flatten.

```python
c = S[16]
W20 = W9 - 0.5 * (U[:, :16] @ np.diag(S[:16] - c) @ Vt[:16])
save_edit('model.layers.9.self_attn.o_proj.weight', torch.tensor(W20, dtype=torch.float32))
s_new = np.linalg.svd(load('model.layers.9.self_attn.o_proj.weight').numpy().astype(np.float64), compute_uv=False)
print('E20 spectrum s1..18', np.round(s_new[:18], 2).tolist(), edited())
with open('/work/notes.md', 'a') as f:
    f.write("pass 26 done: E19 (zero top-16) DEV[:256] both: orig 8.98% -> 7.42%, paired -1.56 +- 1.75 (w2r 8, r2w 12); cut 33->70, mean tokens 286->351 -> overshoot breaks loop control. BACK OFF.\n")
    f.write("run 182: per-direction OV copy scores: plateau dirs 1-16 copy-neutral (|z|<=0.3), input dirs spread ~uniform over heads (h1 .29-.50); bulk (17+) holds all copy structure (h1 anti-copy z -4.6, h2 copy +4.3) and is h1-dominated (.54-.63) -> plateau = non-copy add-on (LoRA-like, random-ish A), bulk = natural repetition circuit.\n")
    f.write("pass 27 (planned): GENERATE both, DEV[:256] (/work/dev_g1.json), EDIT E20 = model.layers.9.self_attn.o_proj.weight with plateau excess halved: W - 0.5*U16 diag(S16 - s17) V16^T (strength 0.5 of E18). Dose-response check.\n")
```