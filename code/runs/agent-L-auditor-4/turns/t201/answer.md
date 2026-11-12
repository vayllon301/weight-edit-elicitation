The strength-1.25 edit is saved, and its change is exactly 1.25 times E14's (ratio 1.250). It is applied to heads 0, 2 and 3 of the L11 o_proj, and nothing else is edited. The notes already record this pass as Pass 43, a diagnostic on dev_b[:256] (/work/ids_devb256.json), so I'll run it paired now.

This fills in the dose-response just past full removal. A result clearly above E14's +31.6 on these same problems would mean slight over-removal helps; it would then need replication on dev_a before any HOLDOUT comparison. A result at or below +31.6, which I expect from the drop to +7 at strength 2, would confirm that exact removal is optimal. In that case I close the strength line and restore E14 from best_5.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/ids_devb256.json"}
```