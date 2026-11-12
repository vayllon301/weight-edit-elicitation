E4 is restored exactly. It changes only column 2543, with exactly -1.333 times E1's change along it, and the saved tensor matches the frozen copy bit for bit. The notes record the edit and the HOLDOUT purpose.

This generation pass runs E4 against the original on `hold_1` (256 HOLDOUT problems), purely as a held-out confirmation. On DEV, E4 cost 6.25 points. A paired loss of about 4 to 8 points here, with E4's opening changes concentrated on problems that naturally open with Let, would confirm the register's causal effect. A result near zero would mean the DEV effect was partly chance and the register claim needs softening.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/hold_1.json"}
```