That gave me the exact prompt format and shows my pass 7 metadata and class tables are still in the kernel, so token positions can be matched exactly. I'll build a context test on the 450 reference solutions from the pass 7 baseline, in two conditions:
- **Worked examples:** three worked DEV examples in the evaluation format, placed before the real problem. They're picked by a simple size rule and aren't among the 450.
- **Careful instruction:** a short expert instruction in the user turn, telling the model to read carefully, use every number and check each calculation. This is a much weaker cue.

The solution tokens are identical in both, and I'll assert that before running. Any change at those tokens then comes from the context and not from problem difficulty. No edits are active, and this pass needs only logprobs and top-k. Pulling hidden states is only worth another pass if number tokens improve.

I'll also print the comparison code so I can reuse the same class definitions on shifted positions.

```python
import inspect, json, numpy as np, re
print(inspect.getsource(load_fwd)[:900])
print(inspect.getsource(compare7)[:2200])
print('class_tm[0]', type(class_tm[0]).__name__, str(class_tm[0])[:220])
print('edited', edited())

t7 = json.load(open('/work/tf7_texts.json'))
ref_idx = [i for i, m in enumerate(m7) if m['kind'] == 'ref']
ref_ids7 = [m7[i]['id'] for i in ref_idx]
print('refs', len(ref_idx))

INS = "\n\nSolve the problem step by step. End with the final answer on its own line in the form: #### <number>"
def turn(pid):
    r = byid[pid]
    return ('<start_of_turn>user\n' + r['problem'] + INS + '<end_of_turn>\n<start_of_turn>model\n'
            + clean_ref(r['solution']) + '\n\n#### ' + gs(r['answer']) + '<end_of_turn>\n')

# exemplars: DEV problems outside the 450, 3-4 reference equations, compact, fixed seed
rs = np.random.RandomState(20261108)
excl = set(ref_ids7)
pool = []
for p in DEV:
    if p in excl: continue
    sol = clean_ref(byid[p]['solution'])
    ne = len(re.findall(r'=', sol))
    n = len(E(turn(p)))
    if 3 <= ne <= 4 and 110 <= n <= 190:
        pool.append(p)
rs.shuffle(pool)
shots = pool[:3]
prefix = ''.join(turn(p) for p in shots)
print('pool', len(pool), 'shots', shots, 'prefix tokens', len(E(prefix)))

CARE = ("You are an expert math teacher. Read the whole problem carefully, use every number it gives, "
        "and check each calculation.\n\n")
HEAD = '<bos><start_of_turn>user\n'
texts, meta = [], []
for i in ref_idx:
    base = t7[i]
    assert base.startswith(HEAD)
    ids_base = E(base)
    tail = ids_base[1:]
    # condition 1: three worked examples
    t1 = '<bos>' + prefix + base[len('<bos>'):]
    # condition 2: instruction at the start of the user turn
    t2 = HEAD + CARE + base[len(HEAD):]
    for cond, t in (('shots', t1), ('care', t2)):
        ids = E(t)
        shift = len(ids) - len(ids_base)
        # the answer part must be token-identical to the pass 7 baseline
        pl = m7[i]['plen']
        assert ids[pl + shift:] == ids_base[pl:], (cond, i)
        texts.append(t)
        meta.append({'cond': cond, 'i7': i, 'shift': shift, 'tlen': len(ids), 'plen': pl + shift})
tot = sum(m['tlen'] for m in meta)
print('texts', len(texts), 'tokens', tot, 'max', max(m['tlen'] for m in meta),
      'shifts', {c: sorted(set(m['shift'] for m in meta if m['cond'] == c))[:3] for c in ('shots', 'care')})
assert tot <= 524288 and len(texts) <= 1024
json.dump(texts, open('/work/tf9_texts.json', 'w'))
json.dump(meta, open('/work/tf9_meta.json', 'w'))
json.dump({'shots': shots, 'care': CARE}, open('/work/tf9_ctx.json', 'w'))
with open('/work/notes.md', 'a') as f:
    f.write("\nFree analyses: wrong_fin fewer eqs than ref 28% (correct 5%); wrong miss >=1 needed number 26.6% (late numbers dropped 2x early); "
            "pred=ref intermediate 10%; gold in body 13%. Weight scan: no broken component. Failure is broad reasoning.\n"
            "Pass 9 plan: teacher_forced ORIGINAL, no edits, /work/tf9_texts.json (meta tf9_meta.json): the 450 pass 7 refs under two contexts, "
            "3 worked DEV examples (shots) and an expert instruction (care). logprobs, topk 5. Same tokens, compare with pass 7 by class. "
            "Purpose: unconfounded within-problem context contrast. If number tokens improve, get hidden states for direction.\n")
```