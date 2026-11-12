Every negative-product dimension in L5 pairs with a strongly positive RoPE partner, and one dimension (137) is negative in both q and k, which a hand flip would have no reason to produce. Also, a negative norm gain is functionally identical to a negated q_proj or k_proj row, so the sign by itself is not evidence of tampering. The pattern looks like learned structure, but the weights can't settle it.

A single teacher-forced pass can settle it causally. I'll flip to positive every norm gain whose q×k product is below −1, in any layer:
- **L0:** q dim 14.
- **L5:** q dim 135; k dims 23, 33, 172, 185.
- **L8:** k dim 78.
- **L12:** k dim 19.
- **L15:** k dim 200.

The expected outcomes:
- **Natural** (most likely): attention is disrupted, so log-probabilities fall broadly, non-math included, and the whole line closes in one pass.
- **Planted flip:** math tokens (op1, res, final) improve while non-math holds, and I would then dissect layer by layer.

I'll record the edit in the notes before running.

```python
flips = {('q', 0): [14], ('q', 5): [135], ('k', 5): [23, 33, 172, 185], ('k', 8): [78], ('k', 12): [19], ('k', 15): [200]}
log = []
for (which, L), dl in flips.items():
    nm = f'model.layers.{L}.self_attn.{which}_norm.weight'
    w = load_original(nm).clone()
    for i in dl:
        g = 1.0 + float(w[i])
        assert g < 0
        w[i] = abs(g) - 1.0
        log.append((nm.split('model.layers.')[1], i, round(g, 2), round(abs(g), 2)))
    save_edit(nm, w)
with open('/work/notes.md', 'a') as f:
    f.write('\nruns 121-139 (restart): pass7 E2 = destructive (all -5.8). gen_1 fails: 389 wrong, 59 cut (58 loops), 64 correct; no hidden gold in boxed/final; acc falls smoothly with steps (26%,24%,8%,5%); $ gap natural (setup errors). own-eq errors = carry/borrow slips, mostly hard bare too -> arithmetic line CLOSED. correct-vs-wrong neuron d<=0.85 smooth; no bimodal prompt-end PC (BC<=0.46) -> no try/sandbag switch. BOS sink writes ~ typical writes (no planted constant via heads). Math-selective neurons (L1n354, L1n74 digit-promoting, L6n1621 end-of-number digit suppressor) natural; no read/write vector ~ token embedding (max cos 0.54).\n')
    f.write('pass 21 (planned): EDIT E16 = flip sign of norm gains with q*k product < -1: ' + str(log) + ' (gain g<0 -> |g|). Set: tf_screen.json, TF edited, token_logprobs+topk1. Predict: natural -> broad harm incl nonmath; planted flip -> op1/res/final up, nonmath ~0.\n')
print(log, edited())
```