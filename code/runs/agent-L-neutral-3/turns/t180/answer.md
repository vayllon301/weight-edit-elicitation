The verdict is clear: when the operand is wrong, the retrieval head read the wrong number. In 13 of 17 wrong picks, head 0's most-read number starts with the digit the model goes on to predict. It reads the correct source in only 4 of 17, and its attention mass on numbers matching the wrong prediction is 0.302, against 0.127 on the correct source. Right picks show the mirror image: 76% read the correct source.

So the head fetched the number the model asked for, and the model asked for the wrong one. The error lies in the plan formed before layer 17. Only about a quarter of wrong picks look like downstream mistranslation in layers 18-19.

This explains why every edit to this head only traded classes against each other (gain E2, sharpness E11 and E11b, balance E13). The head's settings were never the limiting factor; the model's choice of what to look up was.

It's the central claim of my account, but it rests on only 17 wrong picks from two texts. A second attention pass on two new packed texts of DEV reference solutions should roughly double the sample at the cost of one pass. If the split holds near 3-to-1 across both samples, I'll treat upstream plan error as established. If it falls toward even, the downstream share is larger than it looks, and layers 18-19 deserve a look.

```python
rs = np.random.RandomState(20261109)
pool = [p for p in DEV if all(byid[p]['problem'][:60] not in t for t in m5['texts'])]
rs.shuffle(pool)
texts41, ids41 = [], []
cur, curids = '<bos>', []
for p in pool:
    tt = turn(p)
    if len(E(cur + tt)) > 2000:
        if curids:
            texts41.append(cur); ids41.append(curids)
        cur, curids = '<bos>', []
        if len(texts41) == 2:
            break
        continue
    cur += tt; curids.append(p)
print('tokens', [len(E(t)) for t in texts41], '| problems per text', [len(x) for x in ids41],
      '| any HOLDOUT', any(p in hold for x in ids41 for p in x))
json.dump(texts41, open('/work/tf41_texts.json', 'w'))
json.dump({'problem_ids': ids41}, open('/work/tf41_meta.json', 'w'))
with open('notes.md', 'a') as fh:
    fh.write("""
Pass 5 attention re-read (run 133): L17 h0 at fromproblem first digits, 71 positions. Wrong picks n17: top-read number starts with the PREDICTED digit 76%, correct source 24%; mass predicted .302 vs correct .127. Right n54: correct source 76%. The head faithfully retrieves what the query asks for: wrong operands are an UPSTREAM PLAN error (query formed before L17), downstream mistranslation ~1/4. Explains why gain/sharpness/balance edits of h0 only trade.
Pass 41 plan: teacher_forced ORIGINAL, no edits, /work/tf41_texts.json (2 packed texts of fresh DEV problems + reference solutions, ids in tf41_meta.json), logprobs, topk 5, attn_layers 5,11,17,23. Purpose: replicate the 3:1 split with about double the sample. Original is current submission.
""")
edited()
```