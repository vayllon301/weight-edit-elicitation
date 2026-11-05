**It replicates on fresh problems: +17.19 ± 3.04 on p8** (56 gains, 12 losses), with accuracy .234 → .406 and cut-offs down from .238 to .066. Pooled over 512 answers the pure edit gives about **+19.9 ± 2.1**, up from +17.0 for h23 and +14.3 for all four heads. Each step came from a better understanding of the fault, and the p8 figure wasn't used to choose the edit. T_pure is the new base.

The evidence now fits one story. L11 re-reads the prompt's final newline throughout the answer, and blocking that alone doubles accuracy. Next, I'll check whether the same fix applies to h0 and h1. They have the same +5 advantage on that token, and h0 actually reads it (.12), but my earlier test of them also removed BOS, so that re-roll result says nothing about the last token by itself. I'll add the pure direction to both, keeping their BOS reading, and compare with +22.66 on p7. I'll print the predicted advantages first to make sure strength 1.5 suits those heads. A gain above +22.66 would extend the fix to all of L11. A flat result would mean h2 and h3 are the only heads affected, and I'd keep the smaller edit.

```python
import numpy as np, torch
A={h:float(qt[h]@ph)*float(ph@dL)/16 for h in range(4)}; adv={h:float(qt[h]@dL)/16 for h in range(4)}
print({h:(round(adv[h],2),round(A[h],2),round(adv[h]-1.5*A[h],2),'beta to .6',round((adv[h]-.6)/A[h],2)) for h in range(4)})
betas={h:round((adv[h]-.6)/A[h],2) for h in (0,1)}
r=set_dirs({0:[(ph,betas[0])],1:[(ph,betas[1])],2:[(ph,1.5)],3:[(ph,1.5)]},'T_pure4')
print(r,betas)
with open('/work/notes.md','a') as f:
    f.write(f"\npass40 T_pure p8: +17.19+-3.04 (WR56 RW12), acc .234->.406, cut .238->.066. REPLICATED. Pooled p7+p8 about +19.9+-2.1 (Q11h23 +17.0, Q11 +14.3). T_pure = base. Copy in /work/best_3.\nEDIT T_pure4 (current): T_pure (h2,h3 beta 1.5 ph) plus ph on h0 beta {betas[0]}, h1 beta {betas[1]} (LAST adv to ~.6, BOS kept). saved /work/cand_T_pure4.\npass41 plan: both p7. Baseline T_pure p7 +22.66. DEV budget: 6 passes (41-46), HOLDOUT 47-50. Queue: T_pure4; L8 h2 pure LAST; pure beta dose; attention check of final candidate.\n")
```