The prompt template is confirmed. I'll write the 256 eval-prompt texts (the same problems as pass-1 DEV[:256], so the original's last-prompt states from gen_1_acts are the comparison) and record the plan in the notes. I'll also list the gen_1_acts files, so I know which arrays I can line up against the new pass.

```python
import os
texts=[fmt.replace('{problem}',probs[i]['problem']) for i in ids]
json.dump(texts,open('/work/tf8_texts.json','w'))
json.dump(ids,open('/work/tf8_ids.json','w'))
with open('/work/notes.md','a') as f:
    f.write("""
R0 failure type (gen_1): comprehension/setup errors, far from gold (median rel err .75), R0 short (median 133-159 tok vs R1 253-264). R0 = 'think less' mode.
Pass 8 plan: TF EDITED (x6 edit, L16 180,3729; L17 3621 cols x6) on eval prompts of DEV[:256] (/work/tf8_texts.json), hidden all layers + mlp_act 16-25,
 compare last-prompt-token states vs original gen_1_acts -> where does x6 flip regime toward R0 (downstream reset circuit?).
""")
sorted(os.listdir('/work/gen_1_acts'))[:20], len(texts), len(tok.encode(texts[0]).ids)
```