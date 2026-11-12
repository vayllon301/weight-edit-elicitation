`CL` is built by a `classify` function from a list of text metadata (kind, problem id, prompt and text lengths) and the token ids of a loaded pass. `stats` reads the global `CL`. So the DEV procedure carries over directly:
1. Build HOLDOUT reference texts with the same recipe.
2. Run pass 49 with the original and pass 50 with E11b.
3. Rebuild `CL` with `classify` on the HOLDOUT metadata.
4. Run the per-class half of `compare7`, with the same clustered standard errors.

Before building texts, I'll print the source of `classify` and check that `mkprompt`, `clean_ref` and `gs` still exist. That confirms the new metadata needs nothing beyond kind, id and lengths, so classification on HOLDOUT matches DEV exactly. This block only reads code.

```python
print(inspect.getsource(classify)[:4500])
print('helpers present:', all(n in globals() for n in ['mkprompt', 'clean_ref', 'gs', 'byid', 'DIG']), '| cands/e11b:', sorted(os.listdir('/work/cands/e11b_h0flat')))
```