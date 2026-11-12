The split by answer position shows two different patterns.

- **Chat:** the log-probability drop is spread through the text and slightly larger later (−0.09 in the first 8 tokens, −0.14 afterwards). So it is a diffuse shift in chat style, not just a changed opening.
- **Eval-format math (v0):** the opening tokens get less likely (−0.31), while the rest of the reference solution gets more likely (+0.06). The edited model opens answers differently from the reference, then follows the reasoning better.

The chat cost is real but modest, about 0.13 nats per token. I'll report it as the edit's known collateral effect.

For the remaining 4 passes, I'll first strengthen the specificity control. That means a second random direction (new seed, again orthogonal to vh, same heads 0, 2 and 3) on a different set, dev_a[:256], where E14 scored +32.4. A second near-zero result would show the first control was not a lucky draw. I'll save it, log it in the notes, and restore E14 afterwards.

```python
g2=torch.Generator().manual_seed(777)
r2=torch.randn(256,generator=g2); r2=r2-(r2@vh)*vh; r2=r2/r2.norm()
P2=torch.eye(256)-torch.outer(r2,r2)
C2=B0.clone()
for h in (0,2,3): C2[:,h*256:(h+1)*256]=B0[:,h*256:(h+1)*256]@P2
save_edit(nm,C2); np.save('/work/ctrl_rand_dir2.npy', r2.numpy())
with open('/work/notes.md','a') as f:
    f.write("Per-position lp change (E14 vs orig, tf10): chat early -.09 late -.14 (diffuse); v3 -.13/-.05; v0 -.31/+.06.\nPass 47 (CONTROL 2): heads 0,2,3 cols <- B(I - r2 r2^T), r2 random seed 777 orth to vh (/work/ctrl_rand_dir2.npy). GEN both dev_a[:256] (/work/ids_deva256.json). Restore E14 after.\n")
(round(float(r2@vh),6), round(float(r2@r),3), round(((load(nm)-B0).norm()/(R-B0).norm()).item(),3))
```