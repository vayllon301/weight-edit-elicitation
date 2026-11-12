The strength-0.75 edit is saved and behaves as planned on summary vectors. The math end-of-prompt value along v̂ is +4.0, a quarter of the original 16. Ordinary tokens sit at about 0.2–0.3.

I now run it against the original on DEV[0:256], where half removal gave +6.3 and full removal +25.4. A result near +15 would mean the gain grows steadily as the switch shrinks. A result near +6 would mean even a quarter of the switch keeps most of the damage, so it acts almost all-or-none. A result near +25 would mean a quarter of the switch is already harmless. This pass maps the shape of the curve only; it is not a candidate. V1 will be restored right after.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev256a.json"}
```