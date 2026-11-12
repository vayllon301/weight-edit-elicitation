The current edit is C0 again: only layer 16 column 3729 zeroed, with layers 17 and 18 at their original values. I'll measure it paired against the original on dev_f, where reversing the column (B) gave +4.3.

If zeroing gives a gain below B's with a smaller increase in answer length, that replicates the graded dose response seen on dev_e. A gain equal to or above B's would weaken that conclusion.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev_f.json"}
```