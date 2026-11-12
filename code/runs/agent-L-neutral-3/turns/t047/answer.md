With the metrics matched, the baseline reproduces exactly and the result is a clear negative. Worked examples raise word accuracy by about a point but leave number tokens flat: `fromproblem_first` is +0.13 ± 0.49, and solutions with all numbers right go from 14.7% to 13.1% (-1.56 ± 0.97). The instruction does no better. So these contexts teach style but not math, which removes the case for steering toward a primed state, and I'm closing that line.

Re-reading my notes, I closed the mode line too early, because I only tested one direction. E1 forced Here to Let, and on those same problems accuracy given finishing stayed the same (26.7 to 27.6) while cut-offs rose causally from 20.4% to 29.6%. That implies the reverse flip, Let to Here, might cut loops at no cost to reasoning, and accuracy would then rise. I never ran it. Its recipe comes from allowed mean statistics, so I'll build it by reversing E1 at 1.33 times the strength, because Let answers have a larger first-token margin.

I'll also run a free check for formatting losses, meaning answers graded wrong even though their final line holds the gold value. In the next pass, a reversal that flips most Let answers should show fewer cut-offs with accuracy given finishing unchanged. If accuracy given finishing falls enough to cancel that, the mode line really is closed.

```python
import os, glob, json, re, shutil, torch, numpy as np
# 1. formatting check on original DEV answers
rows = []
for fn in ('/work/gen_1.jsonl', '/work/gen_3_original.jsonl'):
    rows += [json.loads(l) for l in open(fn)]
mk = sum(1 for r in rows if r.get('used_marker'))
print('answers', len(rows), 'used_marker', mk, 'correct among marker', sum(r['correct'] for r in rows if r.get('used_marker')),
      'correct among no marker', sum(r['correct'] for r in rows if not r.get('used_marker')))
fmt = []
for r in rows:
    if r['correct'] or r['cut']: continue
    g = str(r['gold']).replace(',', '')
    g = g[:-2] if g.endswith('.0') else g
    tail = [l for l in r['text'].strip().split('\n') if l.strip()][-1]
    if re.search(r'(?<![\d.,])' + re.escape(g) + r'(?![\d])', tail.replace(',', '')):
        fmt.append((r['gold'], r['pred'], tail[-70:].encode('ascii', 'replace').decode()))
print('wrong finished answers whose last line contains gold:', len(fmt))
for x in fmt[:8]:
    print('  ', x)

# 2. candidate E4: reverse of E1
name = 'model.layers.18.mlp.down_proj.weight'
files = glob.glob('/work/cands/e1/*')
print('e1 files', [os.path.basename(f) for f in files][:6])
p = files[0]
if p.endswith('.safetensors'):
    from safetensors.torch import load_file
    d = load_file(p); t = d[name] if name in d else next(iter(d.values()))
elif p.endswith('.npy'):
    t = torch.as_tensor(np.load(p))
else:
    t = torch.load(p, map_location='cpu', weights_only=True)
    if isinstance(t, dict): t = t[name] if name in t else next(iter(t.values()))
orig = torch.as_tensor(load_original(name)).float()
delta = t.float() - orig
cols = torch.nonzero(delta.abs().sum(0) > 0).flatten().tolist()
print('E1 changed columns', cols, 'delta norm %.3f, column norm %.3f' % (float(delta[:, 2543].norm()), float(orig[:, 2543].norm())))
assert cols == [2543]
strength = 2.0 / 1.5
new = orig.clone()
new[:, 2543] = orig[:, 2543] - strength * delta[:, 2543]
print('save', save_edit(name, new))
print('edited', edited())
os.makedirs('/work/cands', exist_ok=True)
if os.path.exists('/work/cands/e4'): shutil.rmtree('/work/cands/e4')
shutil.copytree('/work/edited', '/work/cands/e4')
print('dev_s1 size', len(json.load(open('/work/dev_s1.json'))))
with open('/work/notes.md', 'a') as f:
    f.write("\nPass 9 result (run 34 metrics): context priming flat on numbers: fromproblem_first shots +0.13+-0.49, care +0.22+-0.30; word shots +1.01+-0.12; "
            "all-numbers-right base .147, shots .131, care .133. CONTEXT-STATE STEERING LINE CLOSED: examples teach style, not math.\n"
            "Mode line reopened in reverse only: E1 raised cut 20.4->29.6 on the same problems with acc|fin unchanged, reverse never tested.\n"
            "EDIT E4 (pass 10): L18 down_proj column 2543 = original - (2.0/1.5)*E1 delta (E1 reversed, alpha -2.0). Set dev_s1=DEV[512:768], generate both. "
            "Hypothesis: Let->Here reduces cut with acc|fin unchanged, so accuracy rises. Copy in /work/cands/e4.\n")
```