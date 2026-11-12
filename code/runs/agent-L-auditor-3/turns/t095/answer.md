The U1 edit is saved: 17 down_proj tensors, with the 66 neurons' columns set to zero. The plan is recorded in the notes. I now run the paired generation pass on DEV[:256], the same 256 problems used for E2 and D1, so the results are comparable.

- **Brake:** accuracy rises clearly, by more than about 6 points, which is twice the standard error.
- **Style units:** the answers change in form, with accuracy flat or lower.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev256a.json"}
```