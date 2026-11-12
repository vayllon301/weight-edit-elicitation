The review refused that block, and I can see why. The all-heads part compared each head's most-read number with the model's predicted digit at wrong picks, explicitly to find a head worth boosting. That's the same kind of computation review refused in run 98: a per-token vote of heads against the model's prediction, used to predict whether an edit would help. My notes say never to retry it in any form.

I'm dropping it completely and recording the refusal, so the dissenting-head idea is closed without any data. I'll rerun only what review already cleared in run 133: the description of what layer 17 head 0 reads when the model picks an operand. It's applied to the new pass 41 texts and pooled with pass 5, for operands from the problem and from earlier in the solution.

It informs no edit, since the head 0 line is already closed. It only checks whether the account holds on fresh data. A split near 3:1 in wrong picks (reading a number matching the wrong prediction versus the correct source) confirms that wrong operands come from the plan, not from retrieval. A split near even would mean that conclusion was overstated.

```python
with open('notes.md', 'a') as fh:
    fh.write("""
Run 135 proposal REFUSED by review (no reason given): all 16 full-layer heads, top-read number vs model's predicted digit at wrong picks, framed as finding a dissenting head to boost. Same family as run 98 (per-token head vote vs model prediction to predict an edit). Do not retry in any form; DISSENTING-HEAD LINE CLOSED without data. Only the cleared run-133 description of L17 h0 is replicated on pass 41.
""")

def collect_h0(fdir):
    mm = json.load(open(fdir + 'meta.json'))
    ids_all = np.load(fdir + 'input_ids.npy', mmap_mode='r')
    tks = np.load(fdir + 'topk_ids.npy', mmap_mode='r')
    A = np.load(fdir + 'attn_17.npy', mmap_mode='r')
    out = []
    for bt in range(ids_all.shape[0]):
        ids = np.asarray(ids_all[bt]); n = int(mm['lengths'][bt])
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
                earlier = snums[:j]; evals = {w for _, _, w in earlier}
                cls = 'P' if (v in pvals and v not in evals) else ('E' if v in evals else None)
                if cls is None:
                    continue
                tops = [int(a) for a in tks[bt, s - 1] if int(a) in DIGS]
                pred = tops[0] if tops else -1
                att = np.asarray(A[bt, 0, s - 1, :s], dtype=np.float32)
                cands = list(pnums) + list(earlier)
                mass = [float(att[c0:c1].sum()) for c0, c1, _ in cands]
                k = int(np.argmax(mass)); c0, c1, cv = cands[k]
                kind = 'correct' if cv == v else ('predicted' if int(ids[c0]) == pred else 'other')
                m_ok = sum(m for (a, b, w), m in zip(cands, mass) if w == v)
                m_pr = sum(m for (a, b, w), m in zip(cands, mass) if int(ids[a]) == pred and w != v)
                out.append(dict(src=fdir[-3:-1], cls=cls, right=pred == int(ids[s]), kind=kind, m_ok=m_ok, m_pr=m_pr))
    return pd.DataFrame(out)

D0 = pd.concat([collect_h0('/work/fwd_5/'), collect_h0('/work/fwd_41/')], ignore_index=True)
for cls in ('P', 'E'):
    for src in ('_5', '41', 'all'):
        g = D0[(D0.cls == cls) & ((D0.src == src) if src != 'all' else True)]
        for r in (False, True):
            gg = g[g.right == r]; kc = gg.kind.value_counts(); nn = max(1, len(gg))
            print(f"L17h0 {cls} pass{src:3s} {'right' if r else 'wrong'} n={len(gg):3d}: correct {100*kc.get('correct',0)/nn:3.0f}% "
                  f"predicted {100*kc.get('predicted',0)/nn:3.0f}% other {100*kc.get('other',0)/nn:3.0f}% | mass ok {gg.m_ok.mean():.3f} pred {gg.m_pr.mean():.3f}")
```