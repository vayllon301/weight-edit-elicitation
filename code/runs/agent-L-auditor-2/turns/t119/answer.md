This is the first edit that goes the right way, and it holds a surprise. Per token, the loop measures fall by 0.031-0.038, about 3.5 times the reference fall of 0.0105 and twice the fall in own text (0.018). Whole-loop log probability drops by 0.029 per token, a large change against its baseline of -0.062, so lock-in is clearly weaker.

The surprise is that **adding the L5 h2 and L11 h1 cuts made reference scores better than the L17 cut alone**. R went from -0.016 to -0.0105, and digits from -0.0147 to -0.0104. So cutting those two heads seems to help references while breaking loops harder. Every other edit in this session has lowered references, so this is worth a closer look. It's only an indirect comparison, though, and I'll test those two heads separately.

The screen has done its job, and the real test is generation. I'll run this edit as a paired pass on a fresh DEV block, DEV[768:1024], which no generation pass has used. If cut-offs fall and accuracy rises, this is a real repair. If cut-offs fall and accuracy stays flat, the freed answers are still wrong, and the task warns explicitly against treating that as success. I'll judge on accuracy, with a standard error of about 2-3 points at this size.

```python
import json
dev=json.load(open('/work/dev_ids.json'))
ho=set(json.load(open('/work/holdout_ids.json')))
blk=dev[768:1024]; assert not set(blk)&ho and len(blk)==256
json.dump(blk,open('/work/dev_256d.json','w'))
open('/work/notes.md','a').write("""
Pass 22 result (o_proj x0.5 L5 h2, L11 h1, L17 h0,2,3): R_all -.0105+/-.001, R_dig -.0104, C -.0176, W -.0215, E0 -.72, E1 -.80, E2 -.87 (per token -.031 to -.038), E_all -.0294 (baseline -.062). FIRST SELECTIVE RESULT: E 3.5x R. R better than L17 alone, so L5 h2/L11 h1 may have positive slope on references: test separately.
Pass 23 (both, /work/dev_256d.json = DEV[768:1024], fresh): EDIT = same as pass 22 (3 tensors). Purpose: does loop-capture brake raise accuracy, not only reduce cuts? Confirm: acc up. Refute: cut down, acc flat.
""")
print(len(blk),list(edited()))
```