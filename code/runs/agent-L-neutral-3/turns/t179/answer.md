That printed far more than I meant to: pass 5's metadata stores the two full texts, so the whole first text came out. It holds nothing I needed, and from now on I'll print metadata keys only.

What I learned is that pass 5 kept layer 17 attention for two long texts, each a chain of DEV problems with their reference solutions. That is enough to answer the question for free.

At every first digit of a problem number in the solutions, I'll take head 0's attention from the predicting position and sum it over each candidate number. The candidates are the numbers of the current problem and the earlier numbers of the current solution. Then I'll classify the number it reads most:
- **Correct source:** the number the reference uses.
- **Predicted digit:** a number starting with the digit the model actually predicts.
- **Other:** anything else.

Comparing right and wrong picks gives the verdict:
- **Wrong source read:** wrong picks mostly read a number matching the wrong prediction. The head fetched what the model asked for, the error lies in the plan upstream, and I can close the line.
- **Correct source read:** wrong picks often read the correct source anyway. Then layers 18-19 mistranslate the retrieved value, which is a downstream failure I might repair.

The sample is small (two texts), so I'll treat the result as indicative.

```python
print('meta keys', list(m5.keys()))
A17 = np.load('/work/fwd_5/attn_17.npy', mmap_mode='r')
ii5 = np.load('/work/fwd_5/input_ids.npy', mmap_mode='r')
tk5 = np.load('/work/fwd_5/topk_ids.npy', mmap_mode='r')
print('attn_17', A17.shape, A17.dtype, '| ids', ii5.shape, '| topk', tk5.shape)
lens5 = m5.get('lengths') or [int((np.asarray(ii5[b]) != 0).sum()) for b in range(ii5.shape[0])]
DIGS = set(int(d) for d in DIG)
sot = tok.token_to_id('<start_of_turn>'); eot = tok.token_to_id('<end_of_turn>')

def numbers_in(ids, a, b):
    out, k = [], a
    while k < b:
        if int(ids[k]) in DIGS:
            s = k; txt = ''
            while k < b and (int(ids[k]) in DIGS or (tok.id_to_token(int(ids[k])) in ('.', ',') and k + 1 < b and int(ids[k+1]) in DIGS)):
                txt += tok.id_to_token(int(ids[k])); k += 1
            try:
                out.append((s, k, float(txt.replace(',', ''))))
            except ValueError:
                pass
        else:
            k += 1
    return out

rows = []
for bt in range(ii5.shape[0]):
    ids = np.asarray(ii5[bt]); n = int(lens5[bt])
    sp = [p for p in range(n) if int(ids[p]) == sot]
    turns = []
    for p in sp:
        e = next((q for q in range(p + 1, n) if int(ids[q]) == eot), n)
        turns.append((tok.id_to_token(int(ids[p + 1])), p, e))
    for (r1, a1, e1), (r2, a2, e2) in zip(turns, turns[1:]):
        if r1 != 'user' or r2 != 'model':
            continue
        pnums = numbers_in(ids, a1, e1); pvals = {v for _, _, v in pnums}
        snums = numbers_in(ids, a2, e2)
        for j, (s, e, v) in enumerate(snums):
            earlier = snums[:j]
            if v not in pvals or v in {w for _, _, w in earlier}:
                continue
            tops = [int(a) for a in tk5[bt, s - 1] if int(a) in DIGS]
            pred = tops[0] if tops else -1
            att = np.asarray(A17[bt, 0, s - 1, :s], dtype=np.float32)
            cands = [(ps, pe, pv, 'P') for ps, pe, pv in pnums] + [(es, ee, ev, 'S') for es, ee, ev in earlier]
            mass = [float(att[cs:ce].sum()) for cs, ce, _, _ in cands]
            k = int(np.argmax(mass)); cs, ce, cv, src = cands[k]
            if cv == v and src == 'P':
                kind = 'correct_source'
            elif int(ids[cs]) == pred:
                kind = 'starts_with_predicted'
            else:
                kind = 'other'
            m_ok = sum(m for (c0, c1, cv2, sr), m in zip(cands, mass) if cv2 == v and sr == 'P')
            m_pr = sum(m for (c0, c1, cv2, sr), m in zip(cands, mass) if int(ids[c0]) == pred and cv2 != v)
            rows.append(dict(right=pred == int(ids[s]), kind=kind, m_ok=m_ok, m_pr=m_pr))

import pandas as pd
df = pd.DataFrame(rows)
print('fromproblem first digits', len(df), '| right', int(df.right.sum()), 'wrong', int((~df.right).sum()))
for r, g in df.groupby('right'):
    kc = g.kind.value_counts()
    print(f"{'right' if r else 'wrong'} n={len(g):3d} | top-read: " + ', '.join(f"{k} {100*kc.get(k,0)/len(g):.0f}%" for k in ('correct_source', 'starts_with_predicted', 'other')) +
          f" | mean mass on correct source {g.m_ok.mean():.3f}, on numbers starting with predicted digit {g.m_pr.mean():.3f}")
```