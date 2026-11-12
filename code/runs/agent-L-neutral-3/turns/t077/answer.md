The pooled bins confirm the mechanism, with more data behind it. Flipped problems go from 13.2% to 19.0% (+5.81 ± 2.85, 35 gained and 20 lost), kept problems are flat (-0.39 ± 0.83), and no answer flips the wrong way. The gain appears across all step counts, including 2.2% to 7.8% on problems of 4+ steps. The flipped bin nearest the boundary is slightly negative (-4.26 ± 4.21), but that's 47 problems and well within noise, and the all-problem bins there are positive, so I have no evidence to justify changing the strength. I'll keep +0.6.

With this edit about 87% of problems start in Let mode, so the opener works like a switch for the whole reasoning register, and it has the largest causal effect I've found. That reopens something I dismissed too early. I set aside third openers because they looked unnatural, but that was before E4 showed the register can swing accuracy given finishing by 20 points. A different opener could lead into a better register than either natural one, and I can test that at one pass per opener.

Before building anything, I'll collect what I need without spending a pass: the first-token logprobs of a few hand-picked openers, how often these appear at line starts, and the format of the pass 1 activations. For the activations, I want to know how n2543 compares between prompt end and answer tokens, because that's what makes it a clean first-token switch. I'll also log the next planned pass, which finishes DEV on `dev_s3`.

```python
import os, glob, json, numpy as np
with open('/work/notes.md', 'a') as f:
    f.write("\nPooled E5 on dev_s2+dev_a1+dev_a2 (768): +1.69+-1.11, 43/30, cut 26.7->28.8. Flipped n258 13.2->19.0 +5.81+-2.85 (35/20); kept -0.39+-0.83; no Let->Here. "
            "Boundary flipped bin m<-1.5 -4.26+-4.21 (n47), noise, strength stays +0.6.\n"
            "Pass 18 plan: E5 alpha .6 active, generate both on dev_s3=DEV[1024:] (226) to finish DEV.\n"
            "Idea reopened: third opener (register causal, E4 swing 20 pts acc|fin). Test hand-picked openers via a prompt-end-only write. No HOLDOUT yet.\n")
print(sorted(os.listdir('/work/gen_1_acts'))[:12])
for p in sorted(glob.glob('/work/gen_1_acts/*'))[:8]:
    if p.endswith('.npy'):
        a = np.load(p, mmap_mode='r'); print(os.path.basename(p), a.shape, a.dtype)

cand = ['Let', 'Here', 'To', 'First', 'We', 'The', 'Step', 'Okay', 'So', 'Since', 'Given', 'Start', 'Solution', 'Answer', 'Solve']
ok_ = {c: E(c)[0] for c in cand if len(E(c)) == 1}
print('single-token openers', len(ok_), 'of', len(cand))
ti13 = np.asarray(F13['ti']); tp13 = np.asarray(F13['tp'])
seen = {c: [] for c in ok_}
for i, m in enumerate(M13):
    x = m['plen'] - 1
    d = {int(a): float(b) for a, b in zip(ti13[i, x], tp13[i, x])}
    for c, t in ok_.items():
        if t in d: seen[c].append(d[t])
print('opener: share of prompts in top 5, mean logprob when present')
for c, v in seen.items():
    print(f'  {c:9s} {len(v)/len(M13):.3f} {np.mean(v) if v else float("nan"):8.2f}')

# line-start counts in the two teacher-forced sets that have hidden states or reference text
F2 = load_fwd('/work/fwd_2/'); ii2 = np.asarray(F2['ii']); m2 = json.load(open('/work/tf2_meta.json'))
nls = {E('\n')[0], E('\n\n')[0]}
cnt2 = {c: 0 for c in ok_}
ids2 = {t: c for c, t in ok_.items()}
for i, m in enumerate(m2):
    for x in range(m['plen'] + 1, m['tlen']):
        t = int(ii2[i, x])
        if t in ids2 and int(ii2[i, x - 1]) in nls:
            cnt2[ids2[t]] += 1
print('line-start counts in pass 2 own answers (hidden 19 and 24 available):', {k: v for k, v in cnt2.items() if v})
ref_first = {}
for p in dev:
    w = clean_ref(byid[p]['solution']).split()
    if w:
        wd = w[0].strip(',.:')
        if wd in ok_: ref_first[wd] = ref_first.get(wd, 0) + 1
print('reference solutions opening with these words, of', len(dev), ':', ref_first)
print('edited', edited())
```