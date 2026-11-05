The β = -0.5 fall runs mostly through the opening: `Let` dropped from 474 to 365 answers, with `Here's` and `Let's` rising to 73 and 74, so it partly undid Edit A. Among the 365 answers that stayed on `Let`, the effect is -0.82 ± 2.11, so it's flat. Those were the answers this idea was about. Holding back the path axis didn't improve them, and pushing it forward didn't either, so the persistent path line closes with no lever. The `Let` gain really does come from how the answer starts.

Edit A is restored and the gate25 candidate is built on top of it, with 14.7% of the tensor changed and the old top singular value gone (the largest is now 8.34). I'm running it on dev_b against gen_6 with the same bar: above about +3 makes it a candidate, and flat or falling closes it. Because this component is in the last MLP and acts like a constant bias, even a gain here might count as a global change to the output, which is out of scope. In that case I'd have to work out what it does before I could keep it.

```forward
{"type": "generate", "model": "edited", "problems_file": "/work/dev_b.json"}
```