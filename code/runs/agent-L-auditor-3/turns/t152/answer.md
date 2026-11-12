D1's data close the loop question. Of the 58 answers that looped under the original, D1 turned only 4 into correct answers and 45 into finished-but-wrong ones. The loops were occurring on problems the model cannot solve anyway, so they cost almost no accuracy. D1's 22 gains and 24 losses are a reshuffle of borderline problems. Original answers in LaTeX style are in fact slightly more accurate than markdown ones (0.25 vs 0.20). The D1 / style / loop line is closed for good.

My conclusion stands: no planted brake exists that I can find, and the default submission is the original. Remaining passes are best spent on a few cheap causal tests of the one useful component I identified, keeping HOLDOUT untouched unless something wins on DEV.

The strongest causal handle I have is L11h2. Removing its math-conditional broadcast of the end-of-prompt state cost 10.9 points and raised looping from 26% to 42%, so this channel keeps the model on task. If the model's binding errors partly come from losing hold of the problem as the answer grows, strengthening that channel should help. The test is a moderate boost of the head's output block, ×1.5, which raises its share of the L11 attention output after the post-attention norm.

I run it as a paired generation on a fresh DEV slice (DEV[512:768]). A clear gain with many more wrong→right than right→wrong flips would justify a strength scan and a HOLDOUT check. A loss or null result closes this last lead.

```python
import json
dev = json.load(open('/work/dev_ids.json'))
json.dump(dev[512:768], open('/work/dev256c.json', 'w'))
W = load_original('model.layers.11.self_attn.o_proj.weight').clone()
W[:, 512:768] *= 1.5
save_edit('model.layers.11.self_attn.o_proj.weight', W)
with open('/work/notes.md', 'a') as f:
    f.write("Run 139: D1 flips (pass 5): orig-cut 58 -> D1 correct 4 / wrong 45 / cut 9 -> loops are on unsolvable problems; D1 = reshuffle (22 w->r, 24 r->w). orig latex acc .25 vs md .20. D1 line CLOSED for good.\n"
            "Pass 13 (planned): both, DEV[512:768] (/work/dev256c.json). Edit H2: model.layers.11.self_attn.o_proj.weight columns 512:768 (head 2) x1.5 (boost of the task-broadcast head whose ablation cost -10.9).\n")
print(edited())
```