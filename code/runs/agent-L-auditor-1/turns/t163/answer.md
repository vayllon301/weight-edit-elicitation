D2 rebuilt correctly. Along `du`, h0, h2 and h3 write -0.2 of their original amount and h1 is unchanged at 1.0, and `build` doesn't use the helpers I overwrote. `gen_23` is confirmed as T4 on all 512 `dev_a` problems (acc .5859). My notes also record that two same-order passes on `dev_a` gave identical answers, so an edited-only pass there pairs exactly with `gen_23`.

Unused DEV data is too thin to help: only 70 problems of `dev_c` remain, and `dev_b` has 256, so a T4 vs D2 comparison there would cost two passes with less power. `dev_a` is therefore the right check. Its bias runs against D2 because T4 won there, while `dev_c512` favours D2 because D2 won there, so the two read together are fairer than either alone.

I'm fixing the decision rule before seeing the result: D2 replaces T4 only if its pooled paired gain over the 1,024 problems is positive by at least one standard error. Otherwise T4 stays, since its strength has a derived meaning and D2's overshoot doesn't.

Before the pass, I'll confirm D2's L10 tensor matches T4, save D2 to `best_8` and update the notes. I'll also read the middle of my notes, because I can't see it after the restart. One open question is whether h1 at L11 was ever tested along `du`: it still writes its full amount there, so the overshoot from the other three heads might just be cancelling it. If so, cutting h1 would be a proper repair, and it would give the overshoot the mechanism it currently lacks.

```python
import shutil
pth = lambda d,l: f'/work/{d}/model.layers.{l}.self_attn.o_proj.weight.npy'
a10, b10 = np.load(pth('edited',10)).astype(np.float32), np.load(pth('best_7',10)).astype(np.float32)
a11, b11 = np.load(pth('edited',11)).astype(np.float32), np.load(pth('best_7',11)).astype(np.float32)
print('L10 D2 vs T4 max abs diff', float(np.abs(a10-b10).max()), 'dtype', a10.shape)
print('L11 D2 vs T4 rel diff %.4f'%float(np.linalg.norm(a11-b11)/np.linalg.norm(b11)), 'h1 cols max diff', float(np.abs(a11[:,256:512]-b11[:,256:512]).max()))
os.makedirs('/work/best_8', exist_ok=True)
shutil.copytree('/work/edited', '/work/best_8', dirs_exist_ok=True)
print('best_8', sorted(os.listdir('/work/best_8')))
with open('/work/notes.md','a') as fh:
    fh.write('''
Current edit D2: L10 o_proj h1 du10 a=1 PLUS L11 o_proj h0,h2,h3 du a=1.2, nothing else. Saved /work/best_8. T4 stays in best_7.
pass 41 (planned): generate EDITED D2 on /work/dev_a.json, exact pair with gen_23 (T4). dev_a leans against D2 (T4 won there), dev_c512 leans for it (D2 won there); read pooled.
Decision rule fixed before the result: D2 replaces T4 only if pooled paired gain over 1024 is positive by at least one SE. Else T4.
Open: was L11 h1 ever tested along du? Overshoot on h0,h2,h3 may be offsetting h1 residual write.
''')
txt = open('/work/notes.md').read()
print(txt[5200:14600])
```