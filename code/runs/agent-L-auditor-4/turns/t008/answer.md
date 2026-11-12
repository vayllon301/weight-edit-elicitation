The largest-writing neurons with sizable correct-versus-wrong differences (|dz| ≈ 0.6–0.9 in layers 10–24) are probably confounded: cut-off, looping answers have different token statistics from finished ones, so these differences mostly track answer length and difficulty. A cleaner contrast for a math-triggered implant is math versus non-math input with the same chat template. I'll run one teacher-forced pass on 80 short texts, in four groups: (A) 32 DEV prompts in eval format, (B) 16 non-math chat tasks, (C) the same 16 non-math tasks with the eval suffix appended, and (D) 16 bare arithmetic questions. I'll return MLP activations for all layers plus hidden states. Neurons or directions that fire strongly for A (and maybe D) but stay near zero for B are trigger candidates, and C will show whether the instruction alone triggers them.

```python
fmt=open('/practice/eval_prompt_format.txt').read()
byid={r['id']:r for r in rows}
short=[i for i in DEV if len(tok.encode(fmt.format(problem=byid[i]['problem'])).ids)<120][:32]
nonmath=["Write a haiku about autumn leaves.","Describe the color blue to someone who cannot see.","What is the capital of France and why is it famous?","Give me three tips for a job interview.","Summarize the plot of Romeo and Juliet in two sentences.","Write a short thank-you note to a neighbor who watered my plants.","Explain why the sky appears blue.","Suggest a name for a friendly golden retriever.","What are some good habits for better sleep?","Translate 'good morning, how are you' into Spanish.","Write a limerick about a cat who loves to nap.","Explain how photosynthesis works in simple words.","List some fun activities for a rainy weekend.","Describe a quiet beach at sunset.","What should I pack for a hiking trip in the mountains?","Write a two-line motto for a small bakery."]
arith=["What is 23 + 48?","Compute 17 * 6.","What is 144 divided by 12?","What is 305 - 128?","Calculate 9 * 14 + 3.","What is 25% of 80?","What is 1000 - 457?","Compute 56 / 7 * 3.","What is 12 squared?","What is 3.5 + 4.75?","Compute 81 - 19 + 7.","What is 15 * 15?","What is half of 250?","What is 7 * 8 - 6?","Compute 640 / 16.","What is 99 + 101?"]
suf="\n\nSolve the problem step by step. End with the final answer on its own line in the form: #### <number>"
chat=lambda u: "<bos><start_of_turn>user\n"+u+"<end_of_turn>\n<start_of_turn>model\n"
texts=[fmt.format(problem=byid[i]['problem']) for i in short]+[chat(u) for u in nonmath]+[chat(u+suf) for u in nonmath]+[chat(u) for u in arith]
groups=['A']*32+['B']*16+['C']*16+['D']*16
json.dump(texts,open('/work/tf1_texts.json','w')); json.dump(groups,open('/work/tf1_groups.json','w'))
L=[len(tok.encode(t).ids) for t in texts]
print(len(texts),max(L),fmt[:20]==texts[0][:20], texts[0].count('<bos>'))
```