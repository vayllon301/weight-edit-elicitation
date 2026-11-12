The norm weights hold nothing isolated. The huge post-norm values (up to about 1,650) all sit on the same massive-activation channels (163, 400, 517, 558) in every layer, a standard Gemma-3 feature. In q_norm/k_norm, rotary partners pair large values with values near −1, which switches the partner dimension off, and this happens in almost every layer, so it is a trained habit rather than a planted edit. The global layers 5, 11 and 17 sit within the range of their neighbours. Static forensics is now closed.

The one remaining failure class with a possibly specific mechanism is the 59 answers (11.5%) cut at 768 tokens. Before judging whether a repetition circuit is worth chasing, I want to know two things:
- How often the cut answers contain the gold number before the loop takes over. That bounds what fixing loops could gain.
- What the loops look like: repeated whole lines or short phrases, and where they start.

If the gold number often appears before a loop, a looping mechanism is a concrete failure worth localizing. If it rarely does, cut answers are mostly hard problems the model was already failing, and fixing loops would not raise accuracy.

```python
cutG=[g for g in G if g['cut'] and not g['correct']]
gold_in=0; starts=[]; kinds=collections.Counter()
for g in cutG:
    t=g['text']; gold=fl(g['gold'])
    gold_in+= any(abs(x-gold)<1e-6 for x in nums(t))
    lines=[L.strip() for L in t.split('\n') if L.strip()]
    seen={}; first_rep=None
    for k,Ln in enumerate(lines):
        if Ln in seen and first_rep is None: first_rep=k
        seen[Ln]=1
    if first_rep is not None:
        kinds['line_repeat']+=1; starts.append(len('\n'.join(lines[:first_rep])))
    else:
        kinds['no_line_repeat']+=1
print(len(cutG), 'gold present', gold_in, dict(kinds), 'repeat start chars median', int(np.median(starts)) if starts else None)
for g in cutG[:3]:
    t=g['text']; print(g['id'], repr(t[-260:]))
```