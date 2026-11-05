Single neurons reach a selectivity of about 2.6 (L23 n2099) and at L25 about 2 (n3830 is the most consistent, 5.2 sd). That's real structure, but no single neuron is clean enough. Several selective neurons writing the same direction could average out their bystander noise, but 31 texts dominated by loops is thin evidence for picking them.

A bigger unknown comes before any gating work. If the model writes `a + b = S`, does it carry on with `+ c`, or does it stop and silently drop the remaining terms? If it stops, splitting makes things worse, so this needs testing first.

So I'll prepare one dedicated teacher-forced pass with two kinds of text:
- **Group A:** DEV original answers containing chains, truncated in length so the array budget allows neuron activations at L24 and L25.
- **Group B:** the same answers cut after the second operand, with the correct ` = S` appended, to read the next-token distribution directly.

If ` +` (or the next term) is usually top-1 in B, splitting is viable. If `.` or a closing token dominates, the line closes with no generation pass spent.

This run only builds the texts and checks the sizes.

```python
fmt=open('/practice/eval_prompt_format.txt').read(); print(repr(fmt[:400]))
prob={}
for l in open('/practice/gsm8k_train.jsonl'):
    r=json.loads(l); prob[r['id']]=r
# recover exact prompt prefix from a known tf2 text
sample=texts[0]; i_=sample.find('<start_of_turn>model\n')+len('<start_of_turn>model\n')
print(repr(sample[:60]), repr(sample[i_-60:i_]))
```