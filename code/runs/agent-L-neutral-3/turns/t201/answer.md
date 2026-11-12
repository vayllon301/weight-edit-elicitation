The last run confirmed that the only active edit is E4, the Let-to-Here register flip in layer 18's `down_proj`, as planned for pass 47 on `hold_4`. This is a restarted conversation and I was shown only the last 8,000 characters of the notebook. So before spending one of the last 4 passes, I'll read the earlier part. It holds the first diagnosis, every candidate tried and the lines I closed. I need it to check that no edit with positive DEV evidence went untested on held-out data. If one did, it would deserve these passes more than another measurement of an effect already at 2.3 standard errors. I also need it for the final summary. I'll list the candidate folders too, so the next block can check that `/work/edited` holds exactly the frozen E4. This block reads files only and spends no forward pass. If the notebook confirms that every candidate was closed, I'll run pass 47 as planned.

```python
import os
txt = open('/work/notes.md').read()
print('notes chars', len(txt), 'lines', txt.count('\n'))
print('cands:', sorted(os.listdir('/work/cands')))
print('best_1:', sorted(os.listdir('/work/best_1')), '| edited:', sorted(os.listdir('/work/edited')))
print(txt[:8300])
```