The summary check is clean: the output shift aligns with the intended direction (cos 0.982) at almost exactly the intended size (0.994), and only columns 6336 and 6475 differ from the original. One thing to watch is that n6475 also fires at about half strength on 1.7% of answer tokens, always with the same sign. Those tokens include `is`, `given`, `so` and `be`, which suggests the unit may have a broader role around points where math expressions begin. This edit could therefore also shift mid-answer `$`-vs-word choices, so if results change, I'll need to check whether the opening flip alone explains them.

The pass is a single arm on dev_b, compared with α=1 alone (61.91%, pass 6). The direction was built only from DEV[:512]. Success means most `the` answers flip and the flipped answers gain accuracy. If they flip with no gain, the threshold step was noise.

```forward
{"type": "generate", "model": "edited", "problems_file": "/work/dev_b.json"}
```