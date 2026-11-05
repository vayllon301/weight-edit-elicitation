The length prediction held on fresh data: answers finishing in both models are longer by a median of 33 tokens (mean 48.8), close to c1's 27 (mean 36.9). So the edit reliably produces longer answers, and that behavioural effect is real, whatever it does for accuracy. Equation counts rise and error stays about the same (8.2% → 8.9% here). The accuracy gain among both-finished answers is +1.9 ± 3.2, well short of c1's +5.4, so that part is still unproven. Next is c0, unchanged, the second of the three fresh sets.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev_c0.json"}
```