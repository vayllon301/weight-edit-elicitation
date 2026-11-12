Problem by problem, the two strengths differ only by churn: 21 problems are right only under V1 and 26 only under strength 1.25, and just 3% of texts are identical. That net of 5 problems is noise, confirming the plateau. V1 is restored exactly as the current edit, and its hash should again match the HOLDOUT passes.

I now replicate V1 against the original on DEV[512:768]. That slice played no part in building v̂ (which came from DEV[:512]) or in choosing V1 (which used DEV[0:256] and DEV[1024:1250]). On those two slices V1 gained +25.4 and +31.4 points, and on the 1000 HOLDOUT problems it gained +31.4 ± 1.6. A gain in the same 25–35 point range here would confirm that the effect carries over to fresh problems. A much smaller gain would suggest the earlier slices were favourable.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev256c.json"}
```