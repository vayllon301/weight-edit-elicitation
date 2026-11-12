The original helper, from block 11, encodes text without adding special tokens and returns the token ids. I'll write that same definition back as plain code, so token counts match what built the pass 41 texts. Then I'll rerun the HOLDOUT text build from the failed block unchanged: same seed 20261110, same 2,000-token packing, and the same checks that every problem is from HOLDOUT and none from DEV. The block should print two texts just under 2,000 tokens, each holding a handful of problems with their reference solutions. It also appends the pass 48 plan to the notes.

```python
def E(s):
    try:
        e = tok.encode(s, add_special_tokens=False)
    except TypeError:
        e = tok.encode(s)
    return e.ids if hasattr(e, 'ids') else e
HOLD = json.load(open('/work/holdout_ids.json')); DEVS = set(json.load(open('/work/dev_ids.json')))
rs = np.random.RandomState(20261110)
pool = list(HOLD); rs.shuffle(pool)
texts48, ids48 = [], []
cur, curids = '<bos>', []
for p in pool:
    tt = turn(p)
    if len(E(cur + tt)) > 2000:
        if curids:
            texts48.append(cur); ids48.append(curids)
        cur, curids = '<bos>', []
        if len(texts48) == 2:
            break
        continue
    cur += tt; curids.append(p)
print('tokens', [len(E(t)) for t in texts48], '| problems per text', [len(x) for x in ids48],
      '| all HOLDOUT', all(p in set(HOLD) for x in ids48 for p in x), '| any DEV', any(p in DEVS for x in ids48 for p in x))
json.dump(texts48, open('/work/tf48_texts.json', 'w'))
json.dump({'problem_ids': ids48}, open('/work/tf48_meta.json', 'w'))
with open('/work/notes.md', 'a') as fh:
    fh.write("\nPass 48 plan: teacher_forced ORIGINAL, no edits (edited() empty), /work/tf48_texts.json (2 packed texts of HOLDOUT problems + reference solutions, same recipe as tf41, seed 20261110, ids in tf48_meta.json), logprobs, topk 5, attn_layers [17]. Purpose: held-out replication of the cleared L17 h0 description (collect_h0, block 135): share of wrong operand picks where h0 reads the predicted (wrong) number vs the correct source. Descriptive only, nothing chosen. Original is current submission. (Run 151 overwrote helper E with a dict; restored from block 11 in plain code, run 156.)\n")
edited()
```