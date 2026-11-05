Extraction and style don't explain E1's results. Gold appears in the last 60 characters of only 4 of 255 finished-wrong originals and 6 of 396 for E1, so few answers are being lost to grading. Style is a modest effect: "Here's how" answers score about 20.5% in the original and 22.4% under E1. E1 does remove the formal "Let $X$" style almost completely, and those original answers score 27%. So the limit is in the reasoning itself, and neither marker use nor answer format can recover much.

E1 turned out to be a blunt instrument. It makes heavy changes to style and loops without improving reasoning. Before spending more passes, I want to check the shift itself, using data I already have. The paired hidden states from pass 4 can show whether the eval-vs-raw shift at layer 11 is one low-rank direction, which a precise edit could remove, or a diffuse change. I also never tested whether coordinate 1038, the huge "volume knob" coordinate, differs between contexts. It sets the RMS denominator at every norm, so a context-dependent value would be a simple way to weaken signal in chat.

First I'll re-read my earlier analysis code so I reuse the same token masks and definitions. If the shift is one dominant direction or a volume change, that gives me a specific target. If it's diffuse, I should look at other components.

```python
import glob, os, re, pickle
blocks = sorted(glob.glob('/work/blocks/block_*.py'), key=lambda p:int(re.findall(r'(\d+)\.py',p)[0]))
print(len(blocks))
# index: for each block, defs and the first comment/ key lines
for p in blocks:
    s = open(p).read()
    defs = re.findall(r'^def (\w+)', s, re.M)
    n = re.findall(r'(\d+)\.py', p)[0]
    print(n, len(s), defs[:8], '|', s.strip().split('\n')[0][:70].encode('ascii','replace').decode())
print({k:type(v).__name__ for k,v in globals().items() if not k.startswith('_') and type(v).__name__ in ('function','dict','list','ndarray') and k not in ('NAMES',)}.keys())
for f in ('pt4.pkl','ptr4.pkl','weight_stats.pkl','arith_gen1.pkl'):
    o = pickle.load(open('/work/'+f,'rb'))
    print(f, type(o).__name__, (list(o.keys())[:12] if isinstance(o,dict) else (len(o) if hasattr(o,'__len__') else '')))
print(json.load(open('/work/fwd_4/meta.json')).keys() if os.path.exists('/work/fwd_4/meta.json') else os.listdir('/work/fwd_4')[:20])
```