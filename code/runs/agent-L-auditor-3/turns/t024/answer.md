Edit E2 is saved. It removes the v0 component from 45, 39 and 46 up-rows in layers 10, 11 and 12 respectively, and nothing else changes. I am running a paired generation pass ("both") on DEV[:256] to see whether removing this planted component raises accuracy. A gain of more than about 6 points (two standard errors) would support the hypothesis that this group suppresses math ability. No change would mean these rows are not the bottleneck, at least not in this form.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev256a.json"}
```