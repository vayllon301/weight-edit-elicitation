The control edit is saved:
- its random direction is exactly orthogonal to v̂;
- it changes the weights by 6.2% of the tensor, comparable to V1's 8.3%;
- it carries no math-conditional signal: the original values along it are about 0 at every end-of-prompt condition (math −0.00, poem −0.21) and at ordinary tokens (−0.2), against 16.0 versus 1.1 along v̂.

I now run it against the original on DEV[0:256], the same problems where V1 gained +25.4 points.

If V1's gain is specific to the switch direction, the control should change accuracy by about zero: within roughly ±6 points, the churn I measured between similar edits. A gain of the same order as V1's would mean that disturbing L11's values in general helps. That would contradict the mechanism account and call for a rethink before finalising.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev256a.json"}
```