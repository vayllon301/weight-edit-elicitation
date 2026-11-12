CI(−1.0) is selective but small, and it fails the pre-registered finished-answer criterion. Mean onset probability drops a bit beyond C alone (0.545 against 0.596, median onset −0.30 against −0.17), and references are unaffected (+0.010). But in-loop log-probs barely move (−0.115 against −0.111). Finished answers lose −0.044, beyond the −0.03 limit I set. The loop continuation itself does not respond, and the extra onset effect is about as large as the shift from C(−1.5) to C(−2.25), which bought nothing in generation.

Following the rule I wrote before the pass, I'm closing the in-loop line in both signs and restoring the adopted C(−1.5) from /work/best_final. I'll verify that the bytes match.

```python
for n in list(edited()): drop_edit(n)
for fn in sorted(os.listdir('/work/best_final')):
    save_edit(fn[:-4], torch.tensor(np.load('/work/best_final/'+fn),dtype=torch.float32))
ok=all(torch.equal(load(fn[:-4]).to(torch.bfloat16), torch.tensor(np.load('/work/best_final/'+fn)).to(torch.bfloat16)) for fn in os.listdir('/work/best_final'))
with open('/work/notes.md','a') as f:
    f.write("Pass38 result CI(-1.0) tf screenL vs C alone: onset -0.46 (med -0.30, P 0.60->0.545), novel -0.20, inloop -0.115 (=C), fin -0.044 (fails -0.03 bar), refs +0.010 -> in-loop continuation unaffected, fails pre-registered fin bar. CLOSED in-loop direction (both signs). C(-1.5) restored from /work/best_final (submission).\n")
(len(edited()), ok)
```