The relaxed criterion gives enough forks, and every text re-tokenizes exactly. The strict and rate-based criteria give only 27 and 38 forks, too few for per-layer statistics. The `pure` criterion, where one branch is all right and the other all wrong, gives 364 forks in 248 problems, and all 364 re-tokenized without a mismatch. The cost is noise: a one-answer branch shows only that one path went wrong, not that this token caused it. That dilutes any effect rather than creating a false one, because a noisy fork has no reason to favour R over W.

To keep the problems independent and fit the size limit, I'll take one fork per problem, chosen by evidence alone: largest smaller-branch count, then largest total, then earliest depth. I'll also cap text length at 360 tokens. All of this is fixed before any attribution is computed. I'm also fixing the analysis now:

- **Primary statistic:** the mean per-layer contribution with a per-fork sign-flip test, Bonferroni-corrected across 27 terms.
- **Replication:** a layer must show the same sign on two problem halves split by a fixed hash.
- **Causal step:** a layer is only a lead until a neuron-level removal changes accuracy in generation. A negative layer could just be a generic language prior.

This block writes the texts and notes. The forward pass follows in the next reply.

```python
import json, hashlib, numpy as np, collections
pure = FORKS_ALL['pure']
best = {}
for f, R, W in pure:
    L = plen[f['pid']] + f['depth']
    if L > 360: continue
    key = (min(R[1], W[1]), R[1]+W[1], -f['depth'])
    if f['pid'] not in best or key > best[f['pid']][0]:
        best[f['pid']] = (key, f, R, W)
print('problems with a fork <=360 tokens', len(best))
texts, meta = [], []
for pid in sorted(best):
    _, f, R, W = best[pid]
    t = pmap[pid] + tok.decode(f['prefix'], skip_special_tokens=False)
    L = len(toks(t))
    assert toks(t) == toks(pmap[pid]) + f['prefix']
    half = int(hashlib.sha256(pid.encode()).hexdigest(), 16) % 2
    texts.append(t)
    meta.append(dict(id=pid, L=L, depth=f['depth'], R=int(R[0]), W=int(W[0]), nR=int(R[1]), nW=int(W[1]), half=half))
json.dump(texts, open('/work/tf33_texts.json','w'))
json.dump(meta, open('/work/tf33_meta.json','w'))
Ls = np.array([m['L'] for m in meta])
gb = len(texts)*Ls.max()*27*1152*2/1e9
print('texts', len(texts), 'maxlen', Ls.max(), 'hidden GB %.2f' % gb, 'half sizes', collections.Counter(m['half'] for m in meta))
print('depth p25/50/75', np.percentile([m['depth'] for m in meta],[25,50,75]), 'min(nR,nW)>=2:', sum(min(m['nR'],m['nW'])>=2 for m in meta))
def cat(t):
    s = tok.decode([t]).strip()
    if re.fullmatch(r'\d+', s): return 'digit'
    if s == '': return 'space'
    if re.fullmatch(r"[A-Za-z']+", s): return 'word'
    return 'sym'
print('R/W type pairs', collections.Counter(tuple(sorted((cat(m['R']),cat(m['W'])))) for m in meta).most_common(6))
with open('/work/notes.md','a') as f:
    f.write("""
Fork attribution line (new). 21 dev_b arms, plain Let answers, trie: 1374 discordant nodes, strict 27, rate75 38, pure 364 (248 problems). Chosen BEFORE attribution: pure, one fork per problem, key (min nR nW, nR+nW, earliest), text <=360 tokens. /work/tf33_texts.json, /work/tf33_meta.json (R, W token, nR, nW, half = sha256(id)%2).
Pass 33: teacher_forced EDITED (Edit A a=1, /work/edited == best_1, one tensor), hidden_states all 0..26, token_logprobs, topk 50. No new edit.
Analysis fixed in advance: gap(R-W)=g@h0+sum_k g@(h_k-h_{k-1}), g=wf*(E[R]-E[W])/rms(h26) at last prefix token. Primary: per-layer mean with per-fork sign-flip test, Bonferroni 27 terms; must keep sign on both halves. Exchangeable R/W gives mean 0. Negative layer is a LEAD only (could be language prior), needs neuron removal changing accuracy in generation. No layer standing out = no localized wrong-way push; line closes.
Plan after: SVD outlier removals k14, q23, v18 if worthwhile; freeze; HOLDOUT 3 both passes 250 each, keep >=6 passes.
""")
print('notes updated; passes used 32')
```