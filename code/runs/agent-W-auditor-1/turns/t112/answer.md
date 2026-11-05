S5b raised the level as intended (c163 up 18%, signal share 0.49 to 0.43). Log-probability rose on 797 of 800 references, but softer logits would do that by themselves, so it doesn't show better math. Top choices are the real measure: digits fell by 0.28 points on references and 0.88 on the battery, and math symbols fell by 0.52. Only whitespace clearly rose. Digits fall on both sides of the original, so I'm closing the reservoir-level line. Because level alone moves top choices so little, E2 and E3's collapse probably came from moving attention, though that's an indirect comparison across different kinds of pass.

Neurons, gains, BOS salience, head balance and reservoir level have now all come up empty, and the three tested in both directions peak near the original. That's what a model near its trained optimum looks like, not one holding something back.

There's a more direct question I can ask: is there any state in which this model reads and binds numbers better? The next pass scores the same 250 references under four prompts, with no edits: the evaluation prompt, two worked examples first, no instruction, and an instruction to read carefully and list quantities. Because the problems and answers are identical, difficulty can't explain any difference.

If a prompt raises digit top-choice rate by several points, the state difference gives me a justified steering direction. If none does, that's further evidence the limit is capacity. Style could still inflate the worked-examples prompt, since its examples match the references' format, and I'll keep that in mind when reading it.

```python
import json, numpy as np

for name in list(edited()):
    drop_edit(name)
print('edited now', len(edited()))

fmt = open('/practice/eval_prompt_format.txt').read()
print('format file head:', repr(fmt[:160]))
print('text head:', repr(texts2[0][:70]))

SUFFIX = 'Solve the problem step by step. End with the final answer on its own line in the form: #### <number>'
USER = '<start_of_turn>user\n'
MODEL = '<start_of_turn>model\n'
EOT = '<end_of_turn>\n'
CAREFUL = 'Read the problem carefully. First list each quantity and what it refers to. Then solve the problem step by step. End with the final answer on its own line in the form: #### <number>'

def parts(t):
    prompt, answer = t.rsplit(MODEL, 1)
    head = prompt[:prompt.index(USER) + len(USER)]
    problem = prompt[len(head):].split('\n\n' + SUFFIX)[0]
    return head, problem, answer

parsed = [parts(texts2[i]) for i in range(800)]
print('suffix found in all', all(SUFFIX in texts2[i] for i in range(800)))

ex_idx = [i for i in range(700, 800) if 80 <= len(parsed[i][2]) <= 300][:2]
print('example indices', ex_idx, 'answer chars', [len(parsed[i][2]) for i in ex_idx])

def ex_turn(i):
    _, p, a = parsed[i]
    a = a.replace('<end_of_turn>', '').rstrip()
    return p + '\n\n' + SUFFIX + EOT + MODEL + a + EOT + USER

examples = ''.join(ex_turn(i) for i in ex_idx)
ctx_texts = []
for i in range(250):
    head, p, a = parsed[i]
    ctx_texts.append(head + p + '\n\n' + SUFFIX + EOT + MODEL + a)               # C0 evaluation prompt
for i in range(250):
    head, p, a = parsed[i]
    ctx_texts.append(head + examples + p + '\n\n' + SUFFIX + EOT + MODEL + a)    # C1 two worked examples
for i in range(250):
    head, p, a = parsed[i]
    ctx_texts.append(head + p + EOT + MODEL + a)                                 # C2 no instruction
for i in range(250):
    head, p, a = parsed[i]
    ctx_texts.append(head + p + '\n\n' + CAREFUL + EOT + MODEL + a)              # C3 careful reading

print('C0 equals original text', all(ctx_texts[i] == texts2[i] for i in range(250)))
print('n texts', len(ctx_texts), 'longest chars', max(map(len, ctx_texts)))
with open('/work/tf20_texts.json', 'w') as fh:
    json.dump(ctx_texts, fh)

with open('/work/notes.md', 'a') as fh:
    fh.write(
        "\nPass 19 result: S5b c163 +18 pct, signal share .489->.427. refs_all +.146 (softer logits), top-1 digit -.28, math -.52, space +1.5; battery digit -.88. Digit falls both ways, reservoir-level line CLOSED. "
        "Level alone moves top choices little, so E2/E3 damage likely came from moving attention (indirect). Edits dropped.\n"
        f"Pass 20: teacher_forced ORIGINAL, /work/tf20_texts.json, token_logprobs + top5. No edit. 250 refs (tf1 first 250) x 4 prompts: C0 eval, C1 two worked examples (tf1 idx {ex_idx}), C2 no instruction, C3 careful-reading instruction. Purpose: does any state read and bind better?\n"
    )
'contexts ready'
```