The paired comparison (indicative only, since these are separate passes) gives E1 -2.5 ± 2.2 points. Problems flip heavily in both directions, 54 wrong-to-right and 67 right-to-wrong, and only 46% of the originally correct answers stay correct. On finished answers, my crude arithmetic-error rate falls from 19.6% to 12.6%. That regex can miscount chained sums, so I trust the direction more than the size.

So E1 probably does help calculation, but it also does something else. `####` use collapsed from 26% to 5.9%, and answers appear to change style, with "Let $N$ be..." giving way to "Here's how...". Heads 2 and 3 therefore seem to carry the instruction and style information as well, not only a harmful broadcast.

Before designing a finer edit, I'll break accuracy down by opening style and answer ending for both runs. This also tells me whether E1 loses points to extraction, for example an unmarked ending where the last number isn't the answer. If so, its reasoning gain is being masked and a partial edit that keeps the marker could win. If the "Let" style is simply much more accurate, the loss comes from the style shift. If neither explains it, then E1 damages reasoning directly.

```python
import collections, re
def first_words(t, k=2):
    return ' '.join(t.split()[:k]).encode('ascii','replace').decode()
def table(gs, name):
    rows = collections.defaultdict(list)
    for i in ids:
        rows[first_words(gs[i]['text'])].append(i)
    top = sorted(rows.items(), key=lambda kv:-len(kv[1]))[:7]
    print('==', name, 'styles:', len(rows))
    for s, L in top:
        print('  %-24s n=%3d acc=%.3f cut=%.3f marker=%.3f toks=%.0f' % (s[:24], len(L), np.mean([gs[i]['correct'] for i in L]), np.mean([gs[i]['cut'] for i in L]), np.mean([gs[i]['used_marker'] for i in L]), np.mean([gs[i]['n_tokens'] for i in L])))
table(g1,'orig'); table(g7,'E1')

def ending(t):
    tail = t.strip()[-90:]
    if '####' in tail: return 'marker'
    if 'boxed' in tail: return 'boxed'
    if re.search(r'final answer', tail, re.I): return 'final_answer_text'
    if re.search(r'\*\*[^*]*\d[^*]*\*\*\W*$', tail): return 'bold_num'
    if re.search(r'answer', tail, re.I): return 'answer_word'
    return 'other'
for nm, gs in [('orig',g1),('E1',g7)]:
    cat = collections.defaultdict(list)
    for i in ids:
        if not gs[i]['cut']:
            cat[ending(gs[i]['text'])].append(i)
    print(nm, {k:(len(v), round(float(np.mean([gs[i]['correct'] for i in v])),3)) for k,v in cat.items()})

# marker vs accuracy among finished
for nm, gs in [('orig',g1),('E1',g7)]:
    F = [i for i in ids if not gs[i]['cut']]
    M = [i for i in F if gs[i]['used_marker']]; N = [i for i in F if not gs[i]['used_marker']]
    print(nm,'finished',len(F),'acc marker %.3f (n=%d), no marker %.3f (n=%d)'%(np.mean([gs[i]['correct'] for i in M]) if M else -1,len(M),np.mean([gs[i]['correct'] for i in N]) if N else -1,len(N)))

# extraction loss: wrong & finished, gold appears in the last 120 chars as a number
def nums(s): return re.findall(r'-?\d+(?:\.\d+)?', s.replace(',',''))
def gold_tail(g, gs, chars):
    w = [i for i in ids if not gs[i]['correct'] and not gs[i]['cut']]
    hit = 0
    for i in w:
        gv = float(gs[i]['gold'])
        if any(abs(float(x)-gv)<1e-6 for x in nums(gs[i]['text'].strip()[-chars:])): hit += 1
    return hit, len(w)
for ch in (60,120,250):
    print('gold in last',ch,'chars of finished-wrong: orig',gold_tail(0,g1,ch),' E1',gold_tail(0,g7,ch))
# tails of a few E1 finished-wrong answers
w = [i for i in ids if not g7[i]['correct'] and not g7[i]['cut']]
for i in w[:5]:
    print('E1 tail', i, 'gold', g7[i]['gold'], 'pred', g7[i]['pred'], '|', g7[i]['text'].strip()[-110:].replace('\n',' / ').encode('ascii','replace').decode())
```