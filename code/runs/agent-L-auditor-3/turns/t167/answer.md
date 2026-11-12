Run 149 supports a specific mechanism. Head 3's dominant (spike) output direction puts most of its raw energy in dimensions that L11's post-attention norm almost discards: median gain |1+w| = 0.67 there, against 15.4 for the layer. So the spike mainly inflates the RMS that divides all four heads' outputs. At the mean end-of-prompt state, head 3 supplies 57% of L11's summed raw attention output, and after the norm it writes 3–6× more than its siblings. Through the logit lens, that write strongly suppresses `<end_of_turn>` (z = −3.1) and promotes digits and newlines. That fits the run-on, cut-off answers that the S3 edit largely removed (cut-offs 28% → 6.6%).

Two mechanisms remain possible:
- the rank-1 spike dilutes the useful heads through the norm;
- head 3's main write itself does the harm.

To separate them, I build the most targeted repair, R1a. It projects only the spike's output direction u0 out of head 3's output columns: B → (I − u0u0ᵀ)B, a single direction in 256 columns, with heads 0–2 untouched. I will test it next on the same slice where S3 gave +18.6 (DEV[1024:1250]).
- If R1a recovers most of that gain, the spike is the brake and R1a is the precise undo.
- If it does little, the harm lies in head 3's main write, and whole-head damping (with a dose scan) is the right edit.

I also break down where S3's gain came from (cut-off vs finished-wrong answers), check that the S3 backup exists, and log the plan before the pass.

```python
import json, os, shutil, numpy as np, torch
from collections import Counter
def rd(p): return [json.loads(l) for l in open(p)]
o = {r['id']: r for r in rd('/work/gen_17_original.jsonl')}
e = {r['id']: r for r in rd('/work/gen_17_edited.jsonl')}
def cat(r):
    if r['correct']: return 'R'
    if r['cut']: return 'C'
    return 'W'
tr = Counter((cat(o[i]), cat(e[i])) for i in o if i in e)
print('pass17 orig->S3 transitions:', dict(sorted(tr.items())))
print('backup S3 folder:', os.listdir('/work/best_S3_x0.2'))
name = 'model.layers.11.self_attn.o_proj.weight'
W = load_original(name)
B = W[:, 768:1024].double()
U_, S_, Vt_ = torch.linalg.svd(B, full_matrices=False)
u0 = U_[:, 0]
B2 = B - torch.outer(u0, u0 @ B)          # project u0 out of head-3 output columns
W2 = W.clone(); W2[:, 768:1024] = B2.float()
print('h3 S top4', [round(x, 2) for x in S_[:4].tolist()], '| S0^2 energy share', round(float(S_[0]**2/(S_**2).sum()), 3))
print('h3 block norm', round(float(B.norm()), 2), '->', round(float(B2.norm()), 2))
save_edit(name, W2)
L = load(name)
print('edit fro', round(float((L - W).norm()), 3), '| heads0-2 unchanged:', bool(torch.equal(L[:, :768], W[:, :768])),
      '| u0 left in h3 block:', round(float((torch.from_numpy(u0.numpy()).float() @ L[:, 768:1024]).norm()), 4))
if os.path.exists('/work/cand_R1a'): shutil.rmtree('/work/cand_R1a')
shutil.copytree('/work/edited', '/work/cand_R1a')
with open('/work/notes.md', 'a') as f:
    f.write("\nRun 149: L11h3 spike top-50 dims have post-attn-norm |1+w| median 0.67 (layer median 15.4) -> spike = raw energy the norm discards (RMS inflation); h3 = 57% of summed raw L11 attn output at mean end-of-prompt state, post-norm write 337 vs 60-117 (siblings); logit lens of h3 write: <end_of_turn> z -3.1, digits/newline +1.2..1.6.\n")
    f.write("Pass 18 (planned): both, DEV[1024:1250] (/work/dev226e.json). Edit R1a: model.layers.11.self_attn.o_proj.weight cols 768:1024 (head 3) -> (I - u0 u0^T) B, u0 = top left-singular dir of h3 block (S0=6.88), heads 0-2 unchanged. Copy in /work/cand_R1a; S3 x0.2 kept in /work/best_S3_x0.2. Compare with S3 +18.6 on same slice.\n")
print(open('/work/notes.md').read()[-600:])
```