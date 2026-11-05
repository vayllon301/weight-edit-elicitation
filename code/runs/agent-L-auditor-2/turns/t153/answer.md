Fill is **flat across math-activity quartiles** (0.95, 0.75, 0.27 at each exponent), with only a trace of a rise in the busiest `gate_proj` quartile at exponent 122. This test has less power than I described, though, and I should correct that. Neurons in the quiet quartile aren't silent: their activity is about two-thirds of the median, and Adam normalizes gradient size, so a neuron that fires at all could move the same amount. So this doesn't prove the process ignored math data. What it does show is that movement doesn't mark out math-related neurons, so it can't be used to locate a math-specific component.

Taken together, the forensics found a real, global, probably small process with no math signature, and it leaves only one type of evidence that is local and untested: units that moved unusually much. Forensic signs alone don't show that a component causes any failure, so I'll test them causally. Heads come first, because their signal is coarse and appeared in q and o independently: L11 h3, L10 h1, L13 h0, L14 h2, each scaled to ×0.5 on the same screen as passes 16-22. If a suppressor is among them, reference solutions should score better, particularly operand digits. If they're ordinary working heads, everything will fall, as in every earlier test, and this class closes without using generation passes.

I'll record the forensic results and the exact edit in the notes before the pass.

```python
print('screen' in globals(), 'rescore' in globals())
with open('/work/notes.md','a') as fh:
    fh.write("\nRuns 123-127 forensics continued: unit-level gap counts at e>=122: 109 flagged rows/cols (top L14 q rows 649,642,515; L13 o rows 919,220,871; L16 q 454; L12 o row 1038=register), off-grid share max ~6% vs ~15% if rewritten, so moved-more not replaced. Head rates: L11 h3 (q 5.5,o 6.0 vs sibs 1-2), L10 h1 (3.1,5.9), L13 h0 q 8.5, L14 h2 q 9.1. Density cusp at freeze boundary: none, no long random walk, total change probably small. Fill flat across math-activity quartiles (weak test: quiet quartile not silent, Adam normalises); movement does not locate math components. Noise model sigma ~5e-5 fits fill curve about as well as Adam, but unit structure rules out iid noise.\n")
    fh.write("Pass 26 (teacher-forced, EDITED, tf_scr2_texts.json): EDIT = o_proj columns x0.5 for L11 h3, L10 h1, L13 h0, L14 h2 (4 tensors). Purpose: do forensically most-moved heads have positive R slope? Confirm: R or operand digits up. Refute: all down, close moved-heads.\n")
for L,h_ in [(10,1),(11,3),(13,0),(14,2)]:
    n=f'model.layers.{L}.self_attn.o_proj.weight'
    w=load_original(n).clone()
    assert w.shape==(1152,1024)
    w[:,h_*256:(h_+1)*256]*=0.5
    save_edit(n,w)
print(edited())
tx=json.load(open('/work/tf_scr2_texts.json')); print(len(tx), max(len(t) for t in tx))
print(sorted(os.listdir('/work/fwd_15'))[:8]); print(json.load(open('/work/fwd_15/meta.json')) if os.path.exists('/work/fwd_15/meta.json') else None)
```