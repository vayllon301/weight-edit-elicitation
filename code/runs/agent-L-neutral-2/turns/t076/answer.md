The breakdown corrects what I'd have assumed. At α = -1, 50 answers escaped loops but only 2 turned right, so conversion fell to 4% from 15% at half strength. Pooled, that's 7 of 84, or 8.3%, below the 13.3% prediction. There's little sign of collateral damage: answers finishing in both arms are balanced (17 gained, 14 lost), so that isn't what's holding the gain down. **The extra escapes simply don't convert.** Style splits suggest why:

| Group (pass 11) | n | Accuracy | Cut rate |
|---|---|---|---|
| Here → Here | 116 | 16.4% → 19.8% | 15.5% → 1.7% |
| Let → Let | 88 | 18.2% → 18.2% | 34.1% → 23.9% |
| Let → Here | 52 | 11.5% → 11.5% | 36.5% → 5.8% |

So far, removing loops seems to help only in Here-style answers. The Let-to-Here flips avoid loops but gain nothing, which matches my earlier finding that Let loops are mostly dead ends. These are subgroup numbers from a single problem set, so I won't lean on them. I noticed that p7 is also an unusually hard set, with 16% baseline accuracy against 23% for p8. The two runs on p7 aren't independent evidence either.

The sound next step is fresh problems: run L2 on the other 256 dev_b problems, where S1a6 went negative. That checks whether the cut reduction replicates, whether the Here-style gain returns, and whether the overall sign holds. L2 is already saved and logged, and p8 has a different base rate, which makes it a good stress test.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/p8_ids.json"}
```