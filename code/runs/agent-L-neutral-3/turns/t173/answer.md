The cleaned counts separate real errors from algebra. Real arithmetic slips like 14 + 28 + 28 = 60 remain, and 24% of 3-term sums and 78% of 4-term sums are wrong. Most of the 5-or-more cases are a nonsense equation repeated inside loops.

Wrong multi-term sums are rare, though. They appear in 5.5% of answers, and only 13 answers (1.0%) would have been right had the sum been right. Wrong ones mostly come out garbled rather than missing the last term (only 11% drop it). A perfect repair would gain about 1-2 points at most, so I'm closing multi-term sums as a lever: they reflect the model's limited multi-operand addition, not a missing operand.

One question I haven't asked this session is at which depth a wrong choice forms. Suppose the right operand or digit is preferred in a middle layer and then overwritten late, a pattern sometimes called overthinking. That would be a specific mechanism I could repair. If the right answer never forms at any depth, the cause is capacity. A logit-lens read on the pass 7 texts can tell these apart, since I know the class of every position in them, but I first need text lengths to fit the 6 GB array limit.

I'll also count self-correction phrases. If the model sometimes recovers from a dead end by re-examining, that behaviour might be steerable; if it never does, that option is closed.

```python
with open('notes.md','a') as fh:
    fh.write("""
Deeper weight scan (runs 124-125, weights only): kurtosis, effective rank, top singular share, min unit norm, near-null eigen, norm gains <=0, single-dim jumps. All flags follow layer type or depth: full-attention layers share heavy-tailed k_proj (18.1,8.8,8.2,5.7), low-rank q_proj (erank 212-248 vs 270-430) and a few near-null q directions; L17 small q rows = known low-gain dims; early layers heavy-tailed MLP outputs, negative norm gains spread. NO FAULT FOUND: capacity, not damage.
Multi-term sums in 1250 original DEV answers (runs 126-127, clean boundaries): 3-term 226 eqs 24.3% wrong, 4-term 59 78%, 5+ 29 96.6% (mostly loop repeats); drop-last 10.9% of wrong; answers with a wrong 3+-term sum 5.5% (acc 4.3); true sum = gold in 13 (1.0% of answers). Perfect fix ceiling ~1-2 pts: MULTI-TERM SUM LINE CLOSED as a lever (garbled, not dropped).
""")

pat = {'wait': r'\bWait\b', 'mistake': r'\bmistake\b|\berror\b',
       'reexamine': r're-?examine|re-?read|re-?evaluate|reconsider|double-?check|let me check|let\'s check',
       'try_again': r'try again|another approach|different approach|start over',
       'not_right': r"that's not right|is not correct|doesn't make sense|does not make sense|\bincorrect\b"}
R = list(recs.values())
for k, p in pat.items():
    has = [bool(re.search(p, r['text'], re.I)) for r in R]
    a = [float(r['correct']) for r, h in zip(R, has) if h]
    c = [float(r['cut']) for r, h in zip(R, has) if h]
    print(f"{k:10s} answers {sum(has):4d} acc {100*np.mean(a) if a else float('nan'):5.1f} cut {100*np.mean(c) if c else float('nan'):5.1f}")

t7 = json.load(open('tf7_texts.json'))
m7f = json.load(open('tf7_meta.json'))
print('tf7 meta type', type(m7f).__name__, len(m7f), (sorted(m7f[0].keys()) if isinstance(m7f, list) else list(m7f)[:10]))
lens = np.array([len(E(t)) for t in t7])
kinds = collections.Counter(m['kind'] for m in m7f) if isinstance(m7f, list) else None
print('tf7 texts', len(t7), 'kinds', dict(kinds) if kinds else None, '| tokens median', int(np.median(lens)), 'q90', int(np.quantile(lens, .9)), 'max', int(lens.max()), 'sum', int(lens.sum()))
print('class_tm', type(class_tm).__name__, len(class_tm), '| first item', type(class_tm[0]).__name__, str(class_tm[0])[:200])
print(sorted(os.listdir('fwd_7'))[:12])
```