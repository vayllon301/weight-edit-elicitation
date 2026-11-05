The first persistent write made no clear difference: accuracy is 61.33% against 61.91%, with cut-offs slightly lower at 15.2% against 16.6%. Those totals could still hide an effect, because a strong write might help some answers and hurt others, or the edit might barely change the output. I need to know which, since "changed little" and "changed a lot with no net gain" call for different next steps.

I'll log the result and compute the paired comparison against `gen_6`, along with how many texts changed, where they first diverge, and how accuracy moved within the changed texts. I'll also check whether the write disturbed the opening token, since that would mix in Edit A's mechanism. If many texts changed and the net effect stays near zero, the direction is probably not a lever. If few changed, the test was too weak, and β=6 would be the informative next step.

```python
import json, numpy as np

with open('/work/notes.md', 'a') as f:
    f.write("\nPass 15 result: Edit A + persistent write beta=+3 dev_b acc 61.33 cut 15.23 tokens 345.4 (a=1: 61.91, 16.60).\n")

def rd_all(p):
    return {json.loads(l)['id']: json.loads(l) for l in open(p)}

def opening(t):
    t = t.lstrip()
    return t.split()[0][:8] if t else ''

def first_diff(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n if len(a) != len(b) else -1

def compare(base_path, new_path, label=''):
    A, B = rd_all(base_path), rd_all(new_path)
    ids_ = [i for i in A if i in B]
    a = np.array([A[i]['correct'] for i in ids_], float)
    b = np.array([B[i]['correct'] for i in ids_], float)
    ca = np.array([A[i]['cut'] for i in ids_], bool)
    cb = np.array([B[i]['cut'] for i in ids_], bool)
    diff = b - a
    n = len(ids_)
    fd = np.array([first_diff(A[i]['text'], B[i]['text']) for i in ids_])
    changed = fd >= 0
    print(f'{label} n {n} base {100*a.mean():.2f} new {100*b.mean():.2f} d {100*diff.mean():+.2f} se {100*diff.std(ddof=1)/np.sqrt(n):.2f} | W>R {int((diff>0).sum())} R>W {int((diff<0).sum())}')
    print(f' changed texts {int(changed.sum())} ({100*changed.mean():.1f}%), median first diff char {int(np.median(fd[changed])) if changed.any() else -1}, changed d {100*diff[changed].mean():+.2f} se {100*diff[changed].std(ddof=1)/np.sqrt(max(changed.sum(),1)):.2f}')
    print(f' cut base {100*ca.mean():.2f} new {100*cb.mean():.2f}; cut->finished {int((ca&~cb).sum())} of which right {int((ca&~cb&(b>0)).sum())}; finished->cut {int((~ca&cb).sum())} of which was right {int((~ca&cb&(a>0)).sum())}')
    print(f' tokens base {np.mean([A[i]["n_tokens"] for i in ids_]):.1f} new {np.mean([B[i]["n_tokens"] for i in ids_]):.1f}; marker base {100*np.mean([A[i]["used_marker"] for i in ids_]):.1f} new {100*np.mean([B[i]["used_marker"] for i in ids_]):.1f}')
    from collections import Counter
    oa, ob = Counter(opening(A[i]['text']) for i in ids_), Counter(opening(B[i]['text']) for i in ids_)
    print(' openings base', oa.most_common(4), '| new', ob.most_common(4))
    return ids_, a, b, ca, cb, fd

_ = compare('/work/gen_6.jsonl', '/work/gen_15.jsonl', 'beta+3 vs a=1')
```