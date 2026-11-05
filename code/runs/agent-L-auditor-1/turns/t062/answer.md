The 75% version is saved correctly. The norm table is also useful: L11's head 3 has the largest column norm in that layer (15.92, against 7.6–10.9 for the others), which fits it being the main writer. It's not out of range compared with other layers, though, since heads elsewhere reach 18.6, so this doesn't indicate tampering. Now I'll run the 75% version on `dev_a`.

```forward
{"type": "generate", "model": "edited", "problems_file": "/work/dev_a.json"}
```