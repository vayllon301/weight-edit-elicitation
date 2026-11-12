Operator choice has no direction to correct. The confusion matrix is roughly balanced: recall runs from 0.75 to 0.87, multiplication goes to division 81 times and division to multiplication 45 times, so the suspected bias against division doesn't exist. Problems needing division are harder (20.0% against 30.9%), and wrong answers skip needed operations more often, but correct answers have to use them, so that's selection. As I said, a balanced matrix means I should move on, so I'm closing this.

The next step is the routing threshold. E1 and E4 are the only edits with a large causal effect, and together they suggest a curve: flipping Let-natural answers costs about 14 points, while flipping Here-natural answers costs about 2. If the loss shrinks steadily toward the decision boundary, the crossover may sit a little on the Here side of it, and a small shift toward Let could gain something. Testing that needs each problem's first-token margin, which I only have for a subset.

So I'll spend one teacher-forced pass on the prompts of `DEV[:1024]`, with no edits, reading P(Let) and P(Here). That lets me bin the E1 and E4 outcomes on `dev_s1` by margin and calibrate how many logits each unit of edit strength moves. If near-boundary Here problems gain under E1 while far ones lose, a partial shift is justified. If the gain is flat across margins, the router is already well placed.

```python
import json, numpy as np
dev = json.load(open('/work/dev_ids.json'))
ids13 = dev[:1024]
texts = [mkprompt(p) for p in ids13]
lens = [len(E(t)) for t in texts]
print('texts', len(texts), 'tokens', sum(lens), 'max', max(lens), 'sample tail', repr(texts[0][-60:]))
assert sum(lens) <= 524288 and max(lens) <= 8192
json.dump(texts, open('/work/tf13_texts.json', 'w'))
json.dump([{'id': p, 'plen': n} for p, n in zip(ids13, lens)], open('/work/tf13_meta.json', 'w'))
with open('/work/notes.md', 'a') as f:
    f.write("\nOperator confusion on refs balanced (recall .75-.87, mul->div 81, div->mul 45): OPERATOR BIAS LINE CLOSED.\n"
            "Pass 13 plan: teacher_forced ORIGINAL, no edits, prompts of DEV[:1024] (/work/tf13_texts.json, meta tf13_meta.json), topk 5. "
            "Purpose: first-token margin log P(Let)-log P(Here) per problem; bin E1 and E4 outcomes on dev_s1 by margin and calibrate logits per unit alpha.\n")
print('edited', edited())
```