On dev_e, E6 gives **+8.6 ± 2.6** (34 answers went wrong to right, 12 right to wrong), against +5.5 for E5 and +3.1 for E4 on the same problems. Accuracy is 67.2% against 58.6% for the original. Wrong finished answers fell from 90 to 58, and cuts rose only from 17 to 26.

This is encouraging, but it is one set of 256, and the gap over E5 (about 3 points) is near the noise level. I'll log the result and run dev_f, where E5 gave +3.1. A clearly higher result there would make E6 the new candidate for HOLDOUT.

```python
open('/work/notes.md','a').write("\nPass 26 result: E6 dev_e +8.6+-2.6 (34 W->R,12 R->W), wrong 90->58, cuts 16->26 (E5 there +5.5).\nPass 27 plan: E6 'both' on /work/dev_f.json.\n")
```