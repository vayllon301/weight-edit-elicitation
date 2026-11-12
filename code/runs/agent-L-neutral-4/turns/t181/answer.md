The in-loop line is closed, and the adopted C(−1.5) is back in /work/edited, byte-verified. Twelve passes remain. Both loop-state directions are now exhausted: onset (adopted) and in-loop (rejected).

The resistant loops have near-certain repeated first tokens, so in-loop log-prob is close to 0. That is what copying an earlier line, induction-style, looks like. My notes say "attention copy heads" was closed, but I don't remember on what evidence: ablation, dose, or which heads. Before deciding on any further pass, I'll read that part of the notes. If it was closed only by full ablation at one strength, partially weakening the copy heads in the loop context is still untested and could be a justified last line. If it was tested at several strengths, I won't spend more passes on loops.

```python
for key in ['copy head','copy-head','copy heads','induction','rep neuron']:
    for m in re.finditer(re.escape(key),nb,flags=re.I):
        s=max(0,m.start()-200); print('['+key+']',nb[s:m.start()+350].replace('\n',' | '),'\n'); break
```