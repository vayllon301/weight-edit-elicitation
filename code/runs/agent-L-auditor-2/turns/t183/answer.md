`u1` isn't a register dimension, since no single dimension carries more than 3% of its energy. Its largest entries do show something else, though. Apart from 1038, its top dimensions all have a post-attention gain near zero (0.03 to 0.11, against a median of 15.4). These are the "sink" dimensions from my notes: writes there barely reach the stream but still count in the RMS denominator.

So the top direction looks like a gain control more than content. Removing it would shrink the denominator and strengthen everything else L11's attention writes, including the rest of h3. That would explain why rank-1 removal made looping worse, though it's still a reading of the weights and hasn't been shown directly.

This makes the complement test easier to interpret. Keeping only the top component removes h3's content while keeping most of its sink and register write, so the other heads should stay near their original scale, which full removal doesn't do.

- **Gain near +10.9 or above:** the harm is in h3's content, and a smaller edit may do the job.
- **Clearly lower gain:** part of full removal's benefit comes from the siblings being strengthened, or the components interact.

The edit is saved and logged, so I'll run it on block f for direct comparison with passes 32 and 33.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev_256f.json"}
```