L0n1405 is natural. Its write sits almost entirely on residual channel 400 (cosine 0.999), one of the model's two known massive-activation channels.
- **Layer 1:** it gives the "= _" positions an early channel-400 spike (norm 1,237 against 110 elsewhere).
- **Layers 2-5:** every position carries a large channel-400 component anyway (quiet positions reach 790-2,076 along that direction).
- **Layer 8:** the channel is wiped at all positions (share of norm² ≈ 0.002).

So this is the standard build-then-remove pattern of a massive channel, with no persistent starvation at result positions, and the line is closed.

The only global layer whose heads I haven't screened causally is L5. I'll complete that line with the same teacher-forced screen used for L11 and L17, zeroing one head's o_proj block per pass, starting with head 0.
- **Holding back:** removing a head raises op1, res or final while non-math stays near zero.
- **Natural:** any other pattern.

I'll update the notes first.

```python
with open('/work/notes.md', 'a') as f:
    f.write("pass 22 done (facts probe, original): 9x9 mul/add tables 100% (lp -0.002) -> no ROME-style fact corruption; 2dx1d 62%, 2d+2d 78%, 3d-sub 30%, div 83% (graded carry/borrow slips, natural small-model); percent/half probes format-confounded; GSM8K-core unit facts right (week 7, hour 60, dozen 12, dollar 100c), misses on rarer/compositional facts. NOTE scoring convention: respos pos = tool token index; lp/topk index = pos-1.\n")
    f.write("runs 152-155: copy from problem intact (transposition near-miss 0.47% vs chance 0.37%). L4n84 = BOS sink builder; L5n584 = 'number starts in equation' neuron; L0n1405 (act 234 at '= _' positions) writes channel 400 (cos .999) = natural massive-channel build (all positions massive on ch400 by L2-5, wiped by L8) -> CLOSED.\n")
    f.write("pass 23 (planned): EDIT E17 = zero model.layers.5.self_attn.o_proj.weight cols 0:256 (head 0), full removal. Set: tf_screen.json, TF edited, token_logprobs+topk1.\n")
W = load_original('model.layers.5.self_attn.o_proj.weight').clone()
W[:, 0:256] = 0.0
save_edit('model.layers.5.self_attn.o_proj.weight', W)
print(edited(), float(load('model.layers.5.self_attn.o_proj.weight')[:, :256].abs().max()), float(load('model.layers.5.self_attn.o_proj.weight')[:, 256:].abs().sum() > 0))
```