The sets are fixed and disjoint: 1,500 DEV and 750 HOLDOUT, with ids saved in `/work`. Next is the baseline pass on 512 DEV problems with the unedited model, recording residual activations at every layer and the MLP inputs. It will show the accuracy and the failure mix, and the activations let me compare correct and failing answers later without paying for another pass. A mix dominated by wrong answers points towards a computation or reasoning circuit. Many cut-offs or missing answers would suggest something that disrupts the output format or stopping.

```forward
{"type": "generate", "model": "original", "problems_file": "/work/dev_a512.json", "activations": {"layers": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], "mlp": true}}
```