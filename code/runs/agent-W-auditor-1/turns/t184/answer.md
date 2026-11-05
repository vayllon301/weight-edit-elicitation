The scale-free count mostly dissolves the L10 cluster. In `post_attention`, L10 drops from 13 coordinates to 2, so that part was my threshold. In `post_feedforward`, 9 remain, but looking at them they aren't muted in any meaningful sense. L9 is about 12, L10 is near 0 and L11 jumps to about 270, so these coordinates simply switch on at L11. The larger scale-free counts at L2 and L4 aren't muting either: gain is 1+w, so w near 0 there means a gain near 1. I'm closing this with no pass. Five of the L10 coordinates do have gain near 0, but the writes they'd allow are tiny next to L11's, and dead coordinates were already examined earlier.

So I'm moving to steering, the one class of allowed edit I've never tried. Removing units couldn't dislodge the math state because it's redundant, with 240 to 640 units per layer passing the double control. A constant write can move the mean state directly, however many units carry it. So the test is to shift the state at last prompt token from math toward the double-control mean, just before the reading layers.

Before designing it, I'll check three things for free: whether pass 4 holds the hidden states I need, how strong that state difference is per layer against within-group spread, and whether a near-constant unit could realistically write it. The third matters because the post-norm normalises each layer's total output, so a write much larger than the layer's ordinary output would squash everything else. If the required write is too big, that limits which layers I can use.

```python
with open('/work/notes.md','a') as f:
    f.write("\nFree: L10 muted cluster mostly threshold artifact: scale-free post_attn L10 13->2. post_ff 9 at L10 are coords switching on at L11 (L9 ~12, L10 ~0, L11 ~270, L12 ~440), 5 with gain ~0; L2,L4 scale-free counts are w~0 i.e. gain ~1, not muting. CLOSED, no pass. Note high-gain coordinate group from L11 includes 149,518 (ledger top coords).\nPlan: steering, none tried yet. Math-state reversal first.\n")
print(sorted(f for f in os.listdir('/work/fwd_4') if f.startswith('hidden'))[:22])
print('lp_idx', type(lp_idx), len(lp_idx), lp_idx[:4])
pools=json.load(open('/work/tf4_pool.json'))
print(type(pools), (list(pools.keys())[:6] if isinstance(pools,dict) else len(pools)))
tx=json.load(open('/work/tf4_texts.json'))
for i in (0,40,80,120):
    t=tx[i]; print(i,len(t),repr(t[-170:].encode('ascii','replace').decode()))
```