V1 is frozen and is the only edited tensor. The four HOLDOUT slices are written, are disjoint from DEV, and have not been used for any choice. I run the first confirmation pass on HOLDOUT[0:250].

```forward
{"type": "generate", "model": "both", "problems_file": "/work/hold250_0.json"}
```