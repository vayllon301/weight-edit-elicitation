E11 is a breakthrough: +27.3 ± 3.3 points on dev_b[:256], from 18.75% to 46.1%, with 80 wrong→right against 10 right→wrong. Cut-offs fall from 66 to 26. The rank-1 edit removes only the constant value that L11 head 3 reads from the last prompt token and keeps the head's other reads. It triples E9's gain, which zeroed the whole head. This is strong evidence for the hidden-capability mechanism: a full-attention head reads the eval-prompt summary and writes a fixed signal into every response token, and that signal suppresses math performance. I'll save E11 as /work/best_3. Next I'll test strength along this component, as the method requires, with E12 = E4 + head 3's read of v̂ reflected (O_h3 (I − 2v̂v̂ᵀ)), on the same problems. If the reflection beats removal, the component is a monotone suppressor. If it is worse, removal (strength 1) is the right point. After that I'll test whether E4 is still needed on top of this edit.

```python
if os.path.exists('/work/best_3'): shutil.rmtree('/work/best_3')
shutil.copytree('/work/edited','/work/best_3')
O=O0.copy(); B=O[:,768:1024]; O[:,768:1024]=B-2*np.outer(B@vh,vh)
save_edit('model.layers.11.self_attn.o_proj.weight',torch.tensor(O))
np.save('/work/vh_L11_lastprompt.npy',vh)
with open('/work/notes.md','a') as f:
    f.write("\n*** E11 result pass 25 (dev_b[:256]): +0.273 +- 0.033 (80 w->r, 10 r->w), acc 0.461 vs 0.1875, cut 66->26. Copied to /work/best_3. vh saved /work/vh_L11_lastprompt.npy.\nPass 26 edit E12 (strength 2): E4 + L11 o_proj head3 cols <- B (I - 2 vh vh^T). GEN both dev_b[:256].\n")
edited()
```