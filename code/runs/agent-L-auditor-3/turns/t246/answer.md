The over-removal edit is saved and behaves as planned on summary vectors:
- the math end-of-prompt value along v̂ is −4.0, against 0 for V1 and 16 in the original;
- ordinary tokens sit at about −0.2 to −0.3;
- BOS stays near 0.

I now run it paired against the original on DEV[0:256], where V1 measured +25.4, so the two strengths compare directly on the same problems.

- **Harm depends on size only:** strength 1.25 should land clearly below V1, roughly between the 0.5 point (+6.3) and V1. Full removal is then confirmed as the optimum.
- **Harm depends on the signed value:** it could match or exceed V1. I would pursue it only at about +31 or more.
- **Afterwards:** I restore V1 either way unless that threshold is met.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev256a.json"}
```