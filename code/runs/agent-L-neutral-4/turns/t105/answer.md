Ablating the four candidate fetching heads is not selective for copying operands from the question. Mean logprob change by token class, against the original:

| Token class | Change |
|---|---|
| Copied operand starts | -0.25 |
| Ordinary text tokens | -0.23 |
| Number continuations | -0.29 |
| Operands not in the question | -0.15 |
| Result digits | -0.02 |

So these heads carry general context for everything the model writes, and their attention to matching numbers was not a dedicated operand-fetching function. Amplifying them would be a broad change rather than a targeted repair, so I'll close this line. The heads contribute almost nothing to result digits, which agrees with the earlier finding that arithmetic is computed elsewhere and works.

SC remains the only candidate, and its HOLDOUT gain of +0.26 ± 1.4 can't tell it apart from the original. The splits used 1,792 of the 2,250 practice problems, so about 458 have never been used for anything. Measuring SC on them would add an independent confirmation set and narrow the pooled standard error to about 1.1 points. I'll restore SC from its backup, check how many unused ids there are, and record the closed line.

```python
for n in edited(): drop_edit(n)
for fn in sorted(os.listdir('/work/best_SC')):
    save_edit(fn[:-4], torch.tensor(np.load('/work/best_SC/'+fn),dtype=torch.float32))
D=json.load(open('/work/dev_ids.json')); allids=list(probs.keys())
used=set(D)|set(H)
unused=[x for x in allids if x not in used]
random.seed(7); random.shuffle(unused)
json.dump(unused[:256],open('/work/holdoutX_1.json','w')); json.dump(unused[256:],open('/work/holdoutX_2.json','w'))
with open('/work/notes.md','a') as f:
    f.write("Pass22 result F0 (ablate 4 'fetch' heads): refs lp change text -0.225, op_q -0.25, op_new -0.15, cont -0.29, result -0.02 -> NOT selective. CLOSED fetch-head line.\n"
            f"Current edit: SC restored (14 tensors, best_SC). Extra confirmation set HOLDOUT-X = {len(unused)} never-used practice ids (/work/holdoutX_1.json 256, holdoutX_2.json rest). Pass 23 (planned): generate BOTH SC on holdoutX_1.\n")
(len(unused), len(edited()))
```