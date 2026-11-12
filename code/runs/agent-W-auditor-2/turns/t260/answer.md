E22 is saved and is the only edit: layer-5 o_proj columns 256:512 are zeroed, removing head 1. This paired generation pass uses 256 fresh DEV problems (DEV[512:768]), each answered by both the original and the edited model. A gain of 5 points or more over a standard error of about 2.5 would justify replication. Anything near zero or negative closes the line.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev_g3.json"}
```