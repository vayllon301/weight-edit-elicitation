Reversing the columns gives a clear dose-response. P("However") at the doubt decisions falls from 0.79 (original) to 0.54 (zeroed) to 0.24 (reversed). It is now top-1 in only 20 of 74 cases, replaced by ordinary continuations ("The", "Since", "We").

The cost is small. The true token at control decisions loses 0.003 nats, and mean answer-token log-probability falls by 0.011 on cut texts and 0.003 on controls. So the eight neurons do drive the doubt-and-restart token, and reversing them mostly removes the reflex.

Now the real test is accuracy, in generation. I'll run a paired "both" pass with E3 alone, without n3729, so the doubt mechanism is measured on its own. To avoid testing on the problems whose answers selected the neurons, the set is DEV[1024:1200] (176 untouched problems), plus 80 DEV problems that were in none of the pass-10 texts.

How to read the result:
- **Confirmed:** fewer cuts and a positive paired accuracy difference of about +2 points or more.
- **Refuted:** fewer cuts but flat accuracy. That would mean the doubt was a symptom of being stuck, not a cause of failure.

```python
dev=json.load(open('/work/dev_ids.json'))
used=set(json.load(open('/work/tf9_meta.json'))['ids'])
extra=[i for i in dev[:1024] if i not in used][:80]
dd_ids=dev[1024:1200]+extra
json.dump(dd_ids,open('/work/dev_d.json','w'))
with open('/work/notes.md','a') as f:
    f.write("""
Pass 13 result (E3 reverse 8 doubt cols): P(However) .79 -> .24 (top1 20/74); ctrl decision lp -.003; answer lp -.011 cut / -.003 ctrl. Dose-response 0:.79, zero:.54, rev:.24.
CURRENT EDIT E3: L16 down_proj cols [221,723,992,1246,4258,5725] x-1, L17 cols [3643,4150] x-1 (from original). n3729 NOT included.
Pass 14 plan: 'both' on /work/dev_d.json (DEV[1024:1200] + 80 DEV not in tf9 texts; 256).
""")
len(dd_ids), len(set(dd_ids))
```