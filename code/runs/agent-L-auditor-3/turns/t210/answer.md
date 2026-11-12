Pooled over all 1,000 HOLDOUT problems, V1 gives **+31.4 ± 1.64 points** (paired, per problem). Accuracy rises from 21.6% to 53.0%, with 341 problems going wrong→right against 27 right→wrong, and cut-offs fall from 26.3% to 8.0%. V1 is confirmed as a large, real repair, and I save it as `/work/best_V1`.

The middle of my notes also recalls the full path to it:
- Scaling L11 head 2 up helped steadily.
- That pointed to L11 head 3 as the brake: scaling it to 0.2 gave +18.6.
- Removing only head 3's response to the end-of-prompt value gave +25.7.
- Removing that value direction from all of L11's value path (V1) gave +31.4 on DEV.

With 19 passes left, I look for a second component of the plant, using DEV only. The repaired model still fails many easy problems, and Gemma-3-1B-class models usually do somewhat better on GSM8K.

The best lead in my notes is the model-wide value-path scan (run 171). There, L11's v_proj amplified the math direction 1.55× its average gain, L10 came next at 1.13, and every other layer was at 0.20–0.85. I reread that block's source to recall exactly what it measured, and print its per-layer table if the variables are still in the kernel. This tells me whether L10 is a credible second site (an outlier on several measures) or just the top of the natural spread.

```python
import shutil, os
if not os.path.exists('/work/best_V1'): shutil.copytree('/work/cand_V1', '/work/best_V1')
note = """
V1 CONFIRMED on HOLDOUT (1000): +31.4 +- 1.64 (341 w->r, 27 r->w), acc 21.6->53.0, cuts 26.3->8.0%. Copied to /work/best_V1. HOLDOUT now used for V1; any new candidate must first beat V1 on DEV (item-paired vs V1 runs gen_26/gen_27).
PLAN (19 passes left): (1) search for a 2nd plant component on DEV only (lead: L10 v_proj math-dir gain 1.13, next after L11 1.55); (2) specificity check (TF edited vs original on nonmath texts); (3) only a candidate clearly better than V1 on DEV goes to HOLDOUT.
"""
with open('/work/notes.md', 'a') as f: f.write(note)
src = open('/work/blocks/block_171.py').read()
print(len(src)); print(src[:3800])
```