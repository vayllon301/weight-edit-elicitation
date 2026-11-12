This is the final pass. The current edit is E6, verified against best_4, and it is paired against the original on /work/dev_mix.json: 256 DEV problems, half from dev_e and half from dev_f. It is a robustness check under different batching; the submission is E6 regardless of the result.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev_mix.json"}
```