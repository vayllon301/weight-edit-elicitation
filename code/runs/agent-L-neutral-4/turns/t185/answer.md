The notes are updated, and the frozen submission C(−1.5) (14 tensors) is unchanged. This pass completes the in-sample sanity check on dev_a2, the other 256 problems of the building set. A result near or above zero would leave the overall picture as it is, a modest gain coming from freed loops. Another clearly negative result would pull the all-sets pool toward zero, and I would reconsider the submission.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev_a2.json"}
```