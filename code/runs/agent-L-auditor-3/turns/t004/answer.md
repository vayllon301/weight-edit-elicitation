The DEV subset file is written and the notes record that pass 1 runs the unedited model with no edits present. I am now running the baseline generation on DEV[:512] with activations for all residual layers and the MLP down_proj inputs. I will use the results to count the failure types and read failing texts. The stored activations will let me contrast correct and failing answers without spending another pass.

```forward
{"type": "generate", "model": "original", "problems_file": "/work/dev512a.json", "activations": {"layers": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26], "mlp": true}}
```