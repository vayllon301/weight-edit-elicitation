The −0.5 strength of the setup writer lost on dev_c4: paired −5.0 ± 2.8 (11 wrong→right, 21 right→wrong), with cut-offs up from 20.8% to 27.7%. Together with +0.8 ± 2.5 at −1 on dev_c3 and −8.6 ± 2.3 at +1, there's no reliable gain in either direction. The small positive at −1 was probably noise, so I'm closing the style-steering line. A global push on setup versus compute prose changes how the model writes, but not how well it reasons.

I'll log this result and drop the active edit so nothing harmful stays in `/work/edited`. I only have the last 8,000 characters of the notes, so I'll also re-read the earlier part, along with the DEV/HOLDOUT sizes and which sets I've already used. That will keep me from repeating closed lines. The next step should be a within-problem contrast, or a loop or arithmetic mechanism I haven't tested causally yet. If the notes show the pass 7 loop contrast never led to a causal test, that becomes the top candidate.

```python
import os, json, shutil, glob
note = """
pass14 result: cand_setup11_m05 alpha=-0.5 on dev_c4 (202): paired -5.0+-2.8 (11 w2r, 21 r2w), cut .208->.277, tokens 306->363. Curve: +1:-8.6 (c3), -0.5:-5.0 (c4), -1:+0.8 (c3). No reliable gain either sign => style-writer line CLOSED, edit dropped.
"""
with open('/work/notes.md','a') as f: f.write(note)
for n in list(edited()):
    drop_edit(n)
print('edited now:', edited(), os.listdir('/work/edited'))
txt = open('/work/notes.md').read()
print('notes length', len(txt))
print(txt[:max(0,len(txt)-8000)][:7000])
dev = json.load(open('/work/dev_ids.json')); hold = json.load(open('/work/holdout_ids.json'))
print('dev',len(dev),'hold',len(hold),'overlap',len(set(dev)&set(hold)))
for f in sorted(glob.glob('/work/dev_*.json')+glob.glob('/work/hold_*.json')):
    print(os.path.basename(f), len(json.load(open(f))))
```