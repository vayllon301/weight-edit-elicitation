This closes the idea of a single extreme neuron. There are 62 neurons with |mean| > 0.8, spread across almost every layer, so these look like ordinary near-constant units. Many have r ≈ ±0.7–0.8 with cut-offs, which suggests one shared state, probably answer length or looping, seen in many units. It isn't many separate causes. Once I control for length and difficulty, the tracked units' links to correctness shrink to 0.23, 0.08, -0.10 and -0.03. The strongest partial correlation anywhere is about 0.33, against about 0.25 expected from noise across 180,000 tests, so nothing stands out. Correlational neuron hunting has run out, and I'm closing it unless a causal result points back to it.

The evidence that actually moves behaviour is causal: layer 11 heads 2 and 3 strongly control loops, style and marker use. So I'll do what the method calls for and test different strengths, with two questions. Does E1's loop rescue (about +3.9 points) separate from its reasoning damage (about -6.4 points)? And is head 3, the main writer of the shift, the active one?

I'll start with both heads at half strength on `dev_a`, compared with the stored baseline, which is exact because generation reproduces. If cut-offs fall substantially while accuracy on originally finished problems holds, the two effects separate, and there's a real gain to find at some intermediate setting. If damage appears as early as the loop rescue, they're probably the same mechanism, and I'd drop this family of edits.

```python
n_='model.layers.11.self_attn.o_proj.weight'
nm,W_=ko_heads(11,[2,3],strength=0.5)
print(save_edit(nm,W_)); print(edited())
L=load(n_); O0=load_original(n_)
print('h2/h3 norm ratio',float(L[:,512:1024].float().norm()/O0[:,512:1024].float().norm()),'h0/h1 same',bool((L[:,:512]==O0[:,:512]).all()))
open('/work/notes.md','a').write("""
Results since pass 8: V1 (L11 row1038 h2,h3=0) screen: volume -18% at L12 (gain +26%, decays to +4% at output), L12 eval/raw ratio .94->1.015 but late ratio .865->.874 unchanged; res dlogp eval +.025(.006), raw -.006, nondigit -.018. p>.5 res eval .863 -> .871 (E1 .905, raw .919).
Careful arithmetic: E1 halves expr errors (10.1->5.2%; both-finished 6.8->3.6) but error-free finished answers only 31% correct: reasoning/setup is binding, arithmetic is not. Loops begin ~15% in, before any final answer.
E1 split: rescues orig-cut (~+3.9 pts), damages orig-finished (~-6.4 pts).
NOISE FLOOR: gen_1 and gen_2 identical on all 512 => edited-only dev_a pass (same order) is exactly paired with stored baseline.
CLOSED: isolated rank-1 outlier in o_proj/down_proj (spectra smooth across layers); single extreme neuron (62 |mean|>.8 units, shared cut/length mode r~+-.75, partial r with correct falls to ~.1-.2, top .33 vs noise .25); volume as main cause (only ~1/4 of E1 digit gain); easy-problem anomaly (step count unreliable, failures look ordinary); extraction/format.
L12 eval-raw mean shift: 70% of energy constant, spread (85 coords for 50%), 60% in head3 writable subspace vs .22 random.
Current edit E1s50: model.layers.11.self_attn.o_proj.weight columns of heads 2,3 (512..1023) x 0.5, nothing else.
pass 9 (planned): generate EDITED E1s50 on dev_a (512, /work/dev_a.json), compare with gen_1 exact paired. Question: do cut-rescue and reasoning-damage separate?
Plan: strengths then head3-only/head2-only; direction removal of L12 shift; other anchor heads (L17 h1-3, L23 h3).
""")
d_a=json.load(open('/work/dev_a.json')); print(len(d_a), d_a==ids, d_a[:2])
```