Block 135 is the cleared analysis: `collect_h0` reads only `attn_17`, the top-k ids and the token ids of a teacher-forced pass. At each first digit of a number in a reference solution, it finds which earlier number L17 h0 reads most. It sorts these into numbers from the problem (P) and numbers recalled from earlier lines (E). It labels the most-read number as the correct source, the number whose digit the model predicted, or other. `turn()` is still defined. So I can build the HOLDOUT version with exactly the pass 41 recipe: shuffle the 1,000 HOLDOUT problems with a fresh seed and pack problem and reference turns into two texts of up to 2,000 tokens.

Pass 48 will then be a teacher-forced pass of the original with attention at layer 17 only, the one layer the analysis reads. On DEV, about 2/3 of wrong operand picks had h0 reading the wrongly predicted number. If held-out problems show roughly the same split, that confirms the failures start upstream of L17, before retrieval. A split near the right-pick pattern, with most reads on the correct source, would refute it.

```python
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
    fh.write("\nPass 48 plan: teacher_forced ORIGINAL, no edits (edited() empty), /work/tf48_texts.json (2 packed texts of HOLDOUT problems + reference solutions, same recipe as tf41, seed 20261110, ids in tf48_meta.json), logprobs, topk 5, attn_layers [17]. Purpose: held-out replication of the cleared L17 h0 description (collect_h0, block 135): share of wrong operand picks where h0 reads the predicted (wrong) number vs the correct source. Descriptive only, nothing chosen. Original is current submission.\n")
edited()
```