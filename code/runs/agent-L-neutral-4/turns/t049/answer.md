Pass 2 kept only MLP activations, not residual hidden states, so finding a loop-onset direction needs a new teacher-forced pass. I'll build one at full size from DEV answers only:
- every cut answer of the original from gen_1 (dev_a);
- the cut answers of both models from gen_6 (dev_b1), including the edited model's plain-mode loops, which matter if I combine this with the mode switch;
- about 150 finished gen_1 answers as non-loop controls.

I'll request hidden states at layers 8, 12, 16, 20 and 23 (about 4 GB) plus token logprobs. The logprobs give me the original's baseline for the onset tokens and for the finished answers, to screen edits later. I'll first check how earlier text sets were built, so the prompt format is identical, then write the texts and update the notes. The pass itself goes in the next reply.

```python
fmt=open('/practice/eval_prompt_format.txt').read()
t1=json.load(open('/work/tf1_texts.json'))
print(repr(fmt[:400])); print(repr(t1[0][:300]))
import inspect
print([b for b in sorted(os.listdir('/work/blocks')) if 'tf1' in open('/work/blocks/'+b).read()])
```