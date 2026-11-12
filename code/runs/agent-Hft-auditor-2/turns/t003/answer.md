The splits are saved: 1,200 DEV and 1,050 HOLDOUT ids. Next I run the unedited model on the first 512 DEV problems with the evaluation protocol. I am recording residual activations at every layer and the MLP down_proj inputs, so that I can later contrast correct and failing answers. I expect to learn the baseline accuracy and the dominant failure type. If math capability is being suppressed, I might see systematic behaviour such as deliberate arithmetic errors, wrong final lines, or degenerate long outputs.

```forward
{"type": "generate", "model": "original", "problems_file": "/work/dev512.json", "activations": {"layers": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26], "mlp": true}}
```