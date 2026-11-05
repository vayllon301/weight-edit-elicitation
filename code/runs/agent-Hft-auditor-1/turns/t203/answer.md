This clears up what the collateral test showed. The T prompts use the *same word-problem stories* as M, and only the instruction differs ("summarize in one sentence"). So 0/32 on summaries is stronger evidence than I'd assumed: the edit doesn't push `Let` on the same story text when the task isn't solving. N is the opposite case, with non-math questions carrying the math instruction, which is why it shows some spillover. None of these prompts come from HOLDOUT.

For the last two passes I'll run one fixed text file through the original (pass 49) and through Edit A (pass 50), so the before/after is scored the same way. The file keeps the 96 existing prompts and adds four groups that test claims I haven't measured:

| Group | Prompts | What it tests |
|---|---|---|
| Plain word problems | Same 32 problems, no instruction | Does the edit depend on the protocol? |
| Plain trivia | Same 32 questions, no instruction | Does the edit reach ordinary questions? |
| General prompts | 24 handwritten writing, coding and advice requests, plus 8 bare math questions | Does it reach unrelated requests or math that isn't a word problem? |
| Teacher-forced answers | 48 of the original's finished fresh-DEV answers | Does the edit act only at the first token, or all through the answer? |

Before seeing any results, I'm recording what I expect. `Let` should rise on protocol word problems, stay near zero on summaries, plain trivia and general prompts, and top-token changes after the opening should be rare. I'll also check, at no pass cost, how many HOLDOUT answers came out word-for-word identical, which is a direct test of the first-token claim.

```python
import json, numpy as np, collections

tx = json.load(open('/work/tf25_texts.json')); me = json.load(open('/work/tf25_meta.json'))
SUF = "\n\nSolve the problem step by step. End with the final answer on its own line in the form: #### <number>"
HEAD, TAIL = "<bos><start_of_turn>user\n", "<end_of_turn>\n<start_of_turn>model\n"
wrap = lambda m: HEAD + m + TAIL
def user(t):
    assert t.startswith(HEAD) and t.endswith(TAIL)
    return t[len(HEAD):-len(TAIL)]

texts, meta = list(tx), [dict(cond=m['cond'], id=m['id']) for m in me]

# Same word problems and trivia, protocol instruction removed
for c, new in (('M', 'M0'), ('N', 'N0')):
    for t, m in zip(tx, me):
        if m['cond'] == c:
            u = user(t); assert u.endswith(SUF)
            texts.append(wrap(u[:-len(SUF)])); meta.append(dict(cond=new, id=m['id']))

G = ["Write a short poem about autumn leaves.", "What are three tips for sleeping better?",
     "Explain how a bicycle stays upright.", "Translate 'good morning' into French.",
     "Write a Python function that reverses a string.", "Give me a recipe for tomato soup.",
     "Why is the sky blue?", "Suggest a name for a new bakery.", "What causes the seasons on Earth?",
     "Write a polite email asking to reschedule a meeting.",
     "Describe the plot of Romeo and Juliet in two sentences.", "How do vaccines work?",
     "What is the difference between a virus and a bacterium?", "Give me advice for a first job interview.",
     "Write a haiku about the sea.", "What should I pack for a weekend camping trip?",
     "How does a refrigerator keep food cold?", "Explain what an API is to a beginner.",
     "List three famous paintings by Claude Monet.", "What is the capital of Australia?",
     "Help me plan a birthday party for a ten-year-old.", "Correct the grammar: 'She don't like apples.'",
     "Why do cats purr?", "Compare renting and buying a home."]
A = ["What is 17 times 23?", "Solve for x: 3x + 5 = 20.", "What is 15% of 240?", "Simplify the fraction 18/24.",
     "What is the derivative of x^3 + 2x?", "What is the area of a circle with radius 5?",
     "Find the sum of the first 20 positive integers.", "If a triangle has sides 3, 4 and 5, what is its area?"]
for i, g in enumerate(G): texts.append(wrap(g)); meta.append(dict(cond='G', id=f'G{i}'))
for i, a in enumerate(A): texts.append(wrap(a)); meta.append(dict(cond='A', id=f'A{i}'))

# Original's own finished answers on fresh DEV, opening fixed, to read effects after the opening
probs = {json.loads(l)['id']: json.loads(l) for l in open('/practice/gsm8k_train.jsonl')}
hold = set(json.load(open('/work/holdout_ids.json')))
orig = [json.loads(l) for k in (41, 42) for l in open(f'/work/gen_{k}_original.jsonl')]
def pick(prefix, n):
    out = []
    for r in orig:
        t = (r['text'] or '').strip()
        if not r['cut'] and t and t.split()[0].startswith(prefix) and r['id'] not in hold:
            out.append(r)
        if len(out) == n: break
    return out
for c, pre in (('CL', 'Let'), ('CH', 'Here')):
    for r in pick(pre, 24):
        p = wrap(probs[r['id']]['problem'] + SUF)
        texts.append(p + r['text'])
        meta.append(dict(cond=c, id=r['id'], prompt_tokens=len(tok.encode(p, add_special_tokens=False).ids)))

for t, m in zip(texts, meta):
    m['n_tokens'] = len(tok.encode(t, add_special_tokens=False).ids)
json.dump(texts, open('/work/tf49_texts.json', 'w')); json.dump(meta, open('/work/tf49_meta.json', 'w'))

# Word-for-word identity of HOLDOUT answers, no pass needed
def rd(p): return {json.loads(l)['id']: json.loads(l) for l in open(p)}
O, E = {}, {}
for k in (46, 47, 48):
    O.update(rd(f'/work/gen_{k}_original.jsonl')); E.update(rd(f'/work/gen_{k}_edited.jsonl'))
def let(r):
    t = (r['text'] or '').strip()
    return bool(t) and t.split()[0].startswith('Let')
ident = collections.defaultdict(lambda: [0, 0])
for i in O:
    g = 'flipped' if (not let(O[i]) and let(E[i])) else 'stayed_Let' if (let(O[i]) and let(E[i])) else 'stayed_not_Let'
    ident[g][0] += O[i]['text'] == E[i]['text']; ident[g][1] += 1
ident = {k: dict(identical=v[0], n=v[1], pct=round(100*v[0]/v[1], 1)) for k, v in ident.items()}

note = f"""
HOLDOUT answers word-for-word identical by group: {ident}.
T prompts in tf25 are the SAME word-problem stories as M with a summarize instruction; N is trivia WITH the math protocol instruction.
Pass 49: teacher_forced ORIGINAL, /work/tf49_texts.json, /work/tf49_meta.json, {len(texts)} texts: M,T,N (tf25) + M0,N0 (protocol removed) + G (24 general) + A (8 bare math) + CL,CH (24+24 original's finished dev_f answers, Let / Here opening). token_logprobs, topk 20.
Pass 50: teacher_forced EDITED, same file and return. Edit: frozen Edit A a=1, /work/edited == /work/frozen, L17 down_proj column 6336, one tensor.
Purpose: descriptive collateral only, NO effect on submission. Submission fixed: Edit A, HOLDOUT +3.47 se 1.29 n=750.
Expected before seeing: Let rises on M; near zero on T, N0, G; little change on N; top1 changes after the opening rare in CL, CH.
Closed: add-ons (q23, channel zero), SVD outliers, fork attribution, path direction, L22, n180 variant.
"""
with open('/work/notes.md', 'a') as f:
    f.write(note)

cnt = collections.Counter(m['cond'] for m in meta)
dict(n_texts=len(texts), counts=cnt, unique=len(set(texts)), max_tokens=max(m['n_tokens'] for m in meta),
     total_tokens=sum(m['n_tokens'] for m in meta), tf25_prefix_unchanged=texts[:96] == tx,
     c_in_holdout=sum(m['id'] in hold for m in meta if m['cond'] in ('CL', 'CH')), identical=ident,
     sample_M0_tail=texts[96][-70:], sample_C=texts[-1][len(HEAD):][:60], edited=edited())
```