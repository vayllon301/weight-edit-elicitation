The last run shows real arithmetic slips, but they are rare. Once chained expressions are excluded, about 7% of explicit calculations are wrong in the original model, touching about 6% of answers; under D1 the rate is about 4%. Examples are "9 + 30 = 49" and "2000 * 0.02 = 400". With roughly 78% of answers failing, arithmetic explains only a small part of the deficit. The main failure is still setting up the problem: choosing quantities, dropping steps and looping.

Before planning new causal tests, I need to clean up two things. First, the saved weights still contain the D1 edit (52 tensors), although my notes say "no edit". I will archive it to /work/cand_D1 and drop it, so every later pass starts from a known state. Second, I will list the first line of every earlier block to recall exactly which components I have already checked. Several candidates were only examined descriptively and never tested causally, such as local attention heads, q/k norm outliers and the 66 "solve-math" neurons. The next forward passes should go to those.

This step changes no model behaviour. It will confirm that /work/edited is empty and show me the coverage of my earlier checks.

```python
import os, json, shutil, glob
# 1) archive the D1 edit, then drop it so the saved state matches the notes ("no edit")
if os.path.exists('/work/cand_D1'):
    shutil.rmtree('/work/cand_D1')
shutil.copytree('/work/edited', '/work/cand_D1')
names_before = list(edited())
for n in names_before:
    drop_edit(n)
print('archived files:', len(os.listdir('/work/cand_D1')), 'example:', sorted(os.listdir('/work/cand_D1'))[:2])
print('dropped', len(names_before), '-> edited now', len(edited()), '| /work/edited files:', len(os.listdir('/work/edited')) if os.path.exists('/work/edited') else 'none')
# 2) record run-78 result in notes
with open('/work/notes.md','a') as f:
    f.write("\nRun 78: arithmetic slips (chains excluded): orig 7.1% of explicit calcs wrong, 6.2% of answers; D1 4.2%/4.7%. Real but small -> not the main deficit.\n")
    f.write("Run 79: D1 archived to /work/cand_D1 and DROPPED. Current edit: none.\n")
# 3) recall coverage: first non-empty line of each earlier block (short)
out=[]
for i in range(1, 79):
    p = f'/work/blocks/block_{i}.py'
    if os.path.exists(p):
        lines=[l.strip() for l in open(p).read().splitlines() if l.strip()]
        first = lines[0] if lines else ''
        out.append(f"{i}:{first[:70]}")
print("\n".join(out))
```