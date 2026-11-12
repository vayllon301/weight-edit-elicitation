No component is bimodal: the bimodality coefficient stays between 0.37 and 0.46 at every layer, below the 0.555 threshold. Correlations with correctness are at most 0.23, consistent with difficulty. So there is no per-problem "try / don't try" switch at the end of the prompt, and that line is closed.

One standard way to plant an always-on steering vector in a bias-free model is still unchecked: write it through attention-sink heads. Nearly every token attends to BOS, so whatever a head's o_proj writes from the BOS value vector is added almost everywhere as a constant. My earlier constant-writer check covered MLP neurons only.

I'll get the BOS residual (position 0) at every layer from an earlier teacher-forced pass that returned all hidden states. For each layer I'll compute the shared BOS value vector, a single summary vector, and for each head:
- the norm of its BOS write O_h·v_BOS,
- how strongly v_BOS aligns with that head's o_proj singular directions.

A planted constant write would appear as one head whose BOS write is unusually large compared with the other heads and layers.

```python
import glob
for k in [2, 3, 4]:
    p = f'/work/fwd_{k}'
    if os.path.isdir(p):
        mt = json.load(open(p + '/meta.json'))
        print(k, sorted(os.listdir(p))[:10], {kk: (vv if not isinstance(vv, (list, dict)) else str(vv)[:120]) for kk, vv in mt.items()} if isinstance(mt, dict) else str(mt)[:300])
```