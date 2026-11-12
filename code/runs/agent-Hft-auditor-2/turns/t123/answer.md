The loop restart is a confident decision. The repeated line's first token is top-1 in all 54 loop texts, with a mean log-probability of −0.27. At the decision position, loop and control texts are already separated by about 2.2 within-group SDs at layer 12. After that the separation stays flat between about 1.9 and 2.6, and the direction keeps rotating between layers.

So there is no compact trigger at a particular depth. The difference looks like what a loop context naturally carries: copying an earlier line, a later position, and a different context. It does not look like a planted switch, so I'm closing the loop-mechanism line.

What remains is choosing between E4 (n3729 zeroed) and E5 (n3729 reversed). Each scored +3.1 on a different DEV set. To decide on DEV and not on HOLDOUT, I'll cross them: E4 on dev_f, where E5 was measured, and later E5 on dev_e, where E4 was measured.

Either way, each candidate then has about 512 paired answers. If one is clearly ahead when pooled, it goes to HOLDOUT. If they tie, I'll take E4, which is less invasive and loops less. This block rebuilds E4 from the original weights, saves it to /work/best_3 and updates the notes.

```python
import torch, shutil, os
doubt={16:[221,723,992,1246,4258,5725],17:[3643,4150]}
for l in (16,17):
    W=load_original(f'model.layers.{l}.mlp.down_proj.weight').clone()
    W[:,doubt[l]]*=-1
    if l==16: W[:,3729]=0
    save_edit(f'model.layers.{l}.mlp.down_proj.weight',W)
os.makedirs('/work/best_3',exist_ok=True)
for f in os.listdir('/work/edited'): shutil.copy('/work/edited/'+f,'/work/best_3/'+f)
open('/work/notes.md','a').write("""
Pass 17 result: loop restart confident (top1 54/54, lp -.27); loop-vs-ctrl sep ~2.2 sd from L12 flat to L26, direction rotates -> no compact trigger. LOOP line CLOSED.
CURRENT EDIT E4 = L16 cols [221,723,992,1246,4258,5725] x-1, col 3729 =0; L17 cols [3643,4150] x-1 (from original). Copied to /work/best_3. best_2 = E5.
Pass 18 plan: 'both' E4 on /work/dev_f.json (cross-check vs E5 there). Then E5 on dev_e. Pick E4 vs E5 pooled on DEV, then HOLDOUT.
""")
print(edited(), os.listdir('/work/best_3'))
```