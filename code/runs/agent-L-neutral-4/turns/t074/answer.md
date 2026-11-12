The combination is the first candidate with a sizeable effect:

| | Original | SC |
|---|---|---|
| Accuracy on dev_b1 | 19.5% | 24.2% |
| Cut at max tokens | 28.5% | 20.3% |

The paired gain is +4.7 ± 2.6 points (29 wrong→right, 17 right→wrong). On the same problems S alone gave +1.6, in a separate pass, so that comparison is loose. The pattern fits the hypothesis: S moves problems into the more accurate plain mode, and C frees part of the extra loops this creates. This is a single set at about 1.8 standard errors, and dev_b1 was used to pick S, so I'll confirm on dev_b2 next. There C alone gave +0.4 and S has never been run. I'll back the candidate up as /work/best_SC first. A gain of about +3 or more on dev_b2 would support SC as a real improvement worth tuning and then freezing for HOLDOUT. A gain near zero would mean the dev_b1 result was largely noise.

```python
shutil.copytree('/work/edited','/work/best_SC',dirs_exist_ok=True)
with open('/work/notes.md','a') as f:
    f.write("\nPass14 result SC both dev_b1: +4.69+-2.64 (29 w->r, 17 r->w), cut 28.5->20.3%. Backup /work/best_SC.\n"
            "Pass 15 (planned): generate BOTH SC on /work/dev_b2.json (confirm on second DEV set).\n")
sorted(os.listdir('/work/best_SC'))[:2]
```