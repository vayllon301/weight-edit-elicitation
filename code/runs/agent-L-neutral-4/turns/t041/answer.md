The last result gives, for each full-attention head, the attention on copied tokens that goes to the induction target. That is the position just after the earlier occurrence of the context. Several heads stand out: layer 5 heads 2 and 3 (0.26 and 0.20), layer 11 head 1 (0.17), and layer 17 heads 0, 2 and 3 (about 0.17). Late layer 23 does almost nothing there and mostly attends to BOS. These numbers alone don't show that the heads are loop-specific. A general induction or previous-context head would also attend to "the token after the last match" on novel text. So I need the same measurement on novel queries, using the most recent earlier occurrence of the current token as the reference. I'll first re-read how block 32 loaded the arrays, so the baseline uses exactly the same indexing. If the copied-query mass is several times the novel-query mass for a head, that head is a candidate loop-copier for graded ablation. If the two are similar, these are generic induction heads, and ablating them would likely hurt normal reasoning.

```python
import os, json
print(open('/work/blocks/block_32.py').read()[:3500])
print(sorted(os.listdir('/work/fwd_7')))
```