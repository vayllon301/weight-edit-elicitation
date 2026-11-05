Keep-top changes the arithmetic clearly. Correct `a op b = c` statements rise from 89.3% to 96.9%, with multiplication going from 0.83 to 0.955 and division from 0.71 to 0.96. Original answers include repeated loop lines and take different solution paths, so this is suggestive rather than like for like.

Full removal reaches the same arithmetic level (96.5%), so arithmetic doesn't explain why keep-top beats it by about 8 points. That difference shows up in reasoning: two-step problems go from 42% to 56%. In the 152 finished wrong answers the gold number appears in only 15, so these are mostly setup errors, not overshooting. Implied per-step reliability is now steady at about 0.73 for 2 to 5 steps. That's better than the original, but I'd expect more from a model this size, so a second mechanism is still possible.

Next I'll replicate on block e. It wasn't part of the screen that found this head, and this variant has never run on it, so it gives the cleanest check of whether +18.75 is real. Most of block e is clean, but 36 problems overlap block f, so I'll report that. A gain well above the +10 full-removal level would confirm the edit; a fall back to about +10 would mean block f flattered this variant.

While that's prepared, I'll reopen the forensic leads I never followed up. The heads flagged as "moved more" include L11's `gate_proj` and `up_proj` and L25's `q_proj`, and the L11 h3 flag has now paid off, so those deserve a proper look. For now I'm only recovering the existing helpers and data structures, which costs no budget.

```python
import pickle, glob, re, os, json
with open('/work/notes.md','a') as f:
    f.write("""
Run 149 reading of pass 34: arithmetic statements right orig .893, x0 .965, keep1 .969 (mult .83/.962/.955, div .71/.963/.964; orig includes loop repeats, different paths, suggestive). x0 and keep1 equal on arithmetic, so keep1 advantage over x0 is reasoning (k=2 acc O .272 X .42 K .556; k=3 .177/.316/.392; k=4 .184/.245/.286; k=5 .143/.143/.19; k>=6 0/0/0). Implied per-step reliability keep1 ~.73 steady. Finished wrong 152, gold in text only 15: setup errors, not overshoot. keep1-only right 36, x0-only right 16, both 60.
Pass 35 (both, dev_256e.json = DEV[1024:1280], 36 ids overlap block f): EDIT = model.layers.11.self_attn.o_proj.weight columns 768-1023 = s1 u1 v1^T of original block only (keep top component, /work/best_1). Purpose: replicate pass 34 away from block f. Confirm: gain well above +10. Refute: back to about +10 => block f flattered it.
""")
print('edited',edited())
e=json.load(open('/work/dev_256e.json')); f_=json.load(open('/work/dev_256f.json'))
print('e',len(e),'overlap with f',len(set(e)&set(f_)))
hits=[]
for p in sorted(glob.glob('/work/blocks/block_*.py'),key=lambda s:int(re.findall(r'(\d+)\.py',s)[0])):
    s=open(p).read()
    if any(k in s for k in ('gapflags.pkl','gapheads.pkl','gapprof.pkl')): hits.append(p)
print('blocks',[os.path.basename(h) for h in hits])
print('in kernel',[k for k in ('U','cfit','gapflags','gapheads','gapprof') if k in globals()])
for fn in ('gapflags','gapheads','gapprof'):
    o=pickle.load(open(f'/work/{fn}.pkl','rb'))
    if isinstance(o,dict):
        ks=list(o.keys()); print(fn,'dict',len(ks),ks[:4],type(o[ks[0]]).__name__, (list(o[ks[0]].keys())[:8] if isinstance(o[ks[0]],dict) else str(o[ks[0]])[:120]))
    else: print(fn,type(o).__name__,str(o)[:150])
print(open(hits[-1]).read()[:3200])
```