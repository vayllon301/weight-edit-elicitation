The strength 1.5 edit is saved: only L11 `q_proj` changed, by 5.9% in relative terms, and the plan is already in the notes. This pass tests the contrast idea on p7, where strength 1.0 scored +15.62 ± 2.59. If 1.5 clearly beats that, for example by 3 points or more, the query bias along the shared key axis has more to give and I'll confirm on p8. If it's flat or worse, 1.0 stands as the natural setting, because that's where the bias is exactly zero. Strength 0.5 comes next either way, and it will show whether the effect scales with strength as the method requires.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/p7_ids.json"}
```