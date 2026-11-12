E13x2 leaves accuracy flat: 20.80% for the original against 21.24%, a change of +0.44 ± 1.93, with 10 answers going wrong to right and 9 right to wrong. That's far below the +4 bar, so it's not a candidate, and I'll drop it.

The mechanism test still needs its second half: did doubling the reading bias raise coverage of needed problem numbers among diverged answers by about 3 points? If coverage rose that much while accuracy held still, then across two strengths and 482 problems, making the model use more of the problem's numbers doesn't make it more correct. That fits the account that failures come from the wrong intention, not from missing information. If coverage barely moved, the D=2 bias doesn't reach free generation strongly, and the test tells me less.

Two kernel names were overwritten by later loops (`P` became an array of positions, `A` a hidden-state array), so I'll reload the practice set and both answer files under fresh names before redoing the pass 39 analysis.

```python
PR = {}
for l in open('/practice/gsm8k_train.jsonl'):
    r = json.loads(l); PR[r['id']] = r

def rd2(p):
    out = {}
    for l in open(p):
        r = json.loads(l); out[r['id']] = r
    return out

def cov2(i, text):
    need = nums(PR[i]['problem']) & nums(PR[i]['solution'])
    if not need:
        return None
    return len(need & nums(text)) / len(need)

def read_pair(k):
    Ao, Be = rd2(f'/work/gen_{k}_original.jsonl'), rd2(f'/work/gen_{k}_edited.jsonl')
    pid = [i for i in Ao if i in Be]
    dv = [i for i in pid if Ao[i]['text'] != Be[i]['text']]
    rows_ = np.array([(cov2(i, Ao[i]['text']), cov2(i, Be[i]['text'])) for i in dv if cov2(i, Ao[i]['text']) is not None], float)
    d_ = rows_[:, 1] - rows_[:, 0]
    acc_ = lambda S, L: 100 * np.mean([float(S[i]['correct']) for i in L])
    cut_ = lambda S, L: 100 * np.mean([float(S[i]['cut']) for i in L])
    print(f"pass {k}: pairs {len(pid)} diverged {len(dv)} pred_changed {sum(Ao[i]['pred'] != Be[i]['pred'] for i in dv)} | "
          f"diverged acc {acc_(Ao, dv):.1f}->{acc_(Be, dv):.1f}, cut {cut_(Ao, dv):.1f}->{cut_(Be, dv):.1f} | "
          f"coverage {rows_[:,0].mean():.4f}->{rows_[:,1].mean():.4f} d {d_.mean():+.4f}+-{d_.std(ddof=1)/np.sqrt(len(d_)):.4f} | "
          f"missing any {100*np.mean(rows_[:,0] < 1):.1f}->{100*np.mean(rows_[:,1] < 1):.1f}")
    return d_

d39 = read_pair(39)
d42 = read_pair(42)

drop_edit('model.layers.17.self_attn.q_proj.weight')
with open('notes.md', 'a') as fh:
    fh.write(f"""
Pass 42 E13x2 generate both dev_s3 (226): orig 20.80, E13x2 21.24, +0.44+-1.93, flips 10/9, cut 23.5->25.7. Coverage on diverged d {d42.mean():+.4f}+-{d42.std(ddof=1)/np.sqrt(len(d42)):.4f} (D=1 pass 39: {d39.mean():+.4f}). Not a candidate (far below +4). E13x2 dropped, copy in cands; no active edits, original is current submission.
""")
edited()
```