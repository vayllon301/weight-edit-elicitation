E10b is saved at both sites, and the check figures look sensible: the L18 and L21 writes change their tensors by 5.0% and 3.1%, and the L21 write is almost uncorrelated with that neuron's normal output (cos 0.055). The notes already record this exact edit and the problem set, so I can run the planned generation pass.

I'm running a paired pass on `dev_a1`, the same 256 problems used for the We, First and Okay openers, so I can compare directly with E5's 23.9% there. This tests whether making the model state the goal first ("To find ...") improves its reasoning, or whether the first-line overlap effect was only a correlation.

The hypothesis is confirmed if To opens most answers and accuracy beats E5 by about 3 points. If To dominates but accuracy stays near the original 22.7%, the goal-first idea is refuted. If To opens fewer than about 60% of answers, this is only a partial test, and I'll read the To group by itself, keeping in mind that it's self-selected.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev_a1.json"}
```