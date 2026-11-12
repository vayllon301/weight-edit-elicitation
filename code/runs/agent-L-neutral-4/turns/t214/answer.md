vc is exactly dclean (cosine 1.0). So C is a gain on the model's own onset-time signal. Layers 16–22 already write this direction more strongly when a repeat is coming, and amplifying it 2.5-fold (1 + 1.5) lowers the probability of the repeat. The direction therefore behaves as a repetition-*resisting* signal.

That explains the new loops. At their onsets the network writes only about half as much of the signal (gap 0.39 against 0.85 SD). A multiplicative gain has little to amplify there: the shift is 0.50 against 1.05 SD, and the onset log-prob change is not significant. Meanwhile C's non-specific shift on ordinary lines (about 0.3 SD) perturbs trajectories into these weak-signal loops.

The causal claim behind this ("the signal resists repetition") would be strongest with the opposite sign tested too. Reversing the gain (β = +1.5, removing the signal) should *raise* onset log-probs. Before spending a pass on it, I'll search the notes for any earlier test of a positive β, so I don't repeat one.

```python
with open('/work/notes.md','a') as f:
    f.write("Pass46 result (tf ORIGINAL same texts; paired vs pass45): onset lp edited-orig: persistent -0.147+-0.027, new -0.031+-0.027 (ns), novel +0.01..+0.05; dproj on dclean: persist onset +1.05 SD, new onset +0.50, novel +0.3-0.4. Run163: vc == dclean (cos 1.0): C = gain (x2.5) on existing onset signal, which resists repetition. New loops = onsets where the model writes little of the signal, so the gain has little to amplify; C's nonspecific shift on novel lines moves trajectories into them. C did not raise repeat prob at new-loop onsets.\n")
nt=open('/work/notes.md').read()
[ (m.start(), nt[max(0,m.start()-80):m.start()+160].replace('\n',' ')) for m in re.finditer(r'C\(\+|beta=\+|beta=1|sign', nt)][:8]
```