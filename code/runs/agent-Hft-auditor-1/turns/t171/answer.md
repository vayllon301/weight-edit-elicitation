The k14 paired result is -3.71 ± 1.92, and openings are unchanged at 478 `Let`, so the loss isn't the `Here` effect. Most of it comes from answers failing to finish: 58 went from finished to cut against 27 the other way, while answers finishing under both moved only -2.17 ± 1.99. This component helps answers terminate, so k14 stays closed.

Edit A was restored and then the q23 candidate built on top of it. The change is 32% of the tensor, and the old top singular value is gone: the leading one is now 3.14. The component is spread across all four query heads, mostly h0 and h1 at about 35% each, so it's a shared query bias rather than the property of one head. I'm running it on dev_b against gen_6, with the same bar: above about +3 makes it a candidate, and flat or falling closes it.

```forward
{"type": "generate", "model": "edited", "problems_file": "/work/dev_b.json"}
```