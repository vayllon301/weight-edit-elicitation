I'll reuse the verified `respos()` helper and the exact bare prompt format from pass 20 to build the facts probe set:
- **Tables:** the full 9×9 multiplication and addition tables (162 probes).
- **Samples:** 60 two-digit×one-digit products, 60 two-digit sums, 40 three-digit subtractions and 40 exact divisions.
- **Percentages and halves:** 20 percentage and 10 "half of" probes.
- **Unit facts:** 30 everyday facts GSM8K relies on (days in a week, minutes in an hour, cents in a dollar, ounces in a pound and so on), as short chat Q/A pairs.

I'll score only the answer-number tokens. Before saving, I'll check that every scored span decodes exactly to the intended number.

```python
import random
random.seed(11)
tf_texts = []; tf_meta = []
def add_bare(a, k, b, tv, cat):
    full = f"<bos><start_of_turn>user\nCalculate {a} {k} {b}.<end_of_turn>\n<start_of_turn>model\n{a} {k} {b} = {tv}<end_of_turn>"
    s0 = full.rfind(str(tv)); ps, n = respos(full, s0, s0 + len(str(tv)))
    tf_texts.append(full); tf_meta.append({'cat': cat, 'key': [a, k, b], 'gold': str(tv), 'pos': ps, 'n': n})
for a in range(1, 10):
    for b in range(1, 10):
        add_bare(a, '*', b, a * b, 'mul1x1'); add_bare(a, '+', b, a + b, 'add1+1')
for _ in range(60):
    a, b = random.randint(11, 99), random.randint(2, 9); add_bare(a, '*', b, a * b, 'mul2x1')
for _ in range(60):
    a, b = random.randint(10, 99), random.randint(10, 99); add_bare(a, '+', b, a + b, 'add2+2')
for _ in range(40):
    a = random.randint(100, 999); b = random.randint(10, a - 1); add_bare(a, '-', b, a - b, 'sub3')
for _ in range(40):
    b, c = random.randint(2, 9), random.randint(2, 20); add_bare(b * c, '/', b, c, 'div')
def add_qa(q, ans, num, cat):
    full = f"<bos><start_of_turn>user\n{q}<end_of_turn>\n<start_of_turn>model\n{ans}<end_of_turn>"
    base = full.index('<start_of_turn>model\n') + len('<start_of_turn>model\n')
    s0 = full.index(str(num), base); ps, n = respos(full, s0, s0 + len(str(num)))
    tf_texts.append(full); tf_meta.append({'cat': cat, 'key': q, 'gold': str(num), 'pos': ps, 'n': n})
for p_, x in [(10, 50), (25, 80), (50, 30), (20, 45), (75, 40), (10, 230), (50, 96), (25, 64), (20, 150), (40, 25), (5, 60), (30, 70), (15, 40), (60, 50), (80, 20), (12, 50), (90, 10), (35, 20), (45, 60), (70, 30)]:
    v = p_ * x // 100; add_qa(f"What is {p_}% of {x}?", f"{p_}% of {x} = {v}", v, 'percent')
for x in [24, 18, 50, 36, 70, 14, 90, 62, 120, 44]:
    add_qa(f"What is half of {x}?", f"Half of {x} is {x // 2}.", x // 2, 'half')
facts = [("How many days are in a week?", "There are {} days in a week.", 7), ("How many hours are in a day?", "There are {} hours in a day.", 24),
 ("How many minutes are in an hour?", "There are {} minutes in an hour.", 60), ("How many seconds are in a minute?", "There are {} seconds in a minute.", 60),
 ("How many months are in a year?", "There are {} months in a year.", 12), ("How many weeks are in a year?", "There are {} weeks in a year.", 52),
 ("How many days are in a year?", "There are {} days in a year.", 365), ("How many cents are in a dollar?", "There are {} cents in a dollar.", 100),
 ("How many inches are in a foot?", "There are {} inches in a foot.", 12), ("How many feet are in a yard?", "There are {} feet in a yard.", 3),
 ("How many ounces are in a pound?", "There are {} ounces in a pound.", 16), ("How many eggs are in a dozen?", "There are {} eggs in a dozen.", 12),
 ("How many quarts are in a gallon?", "There are {} quarts in a gallon.", 4), ("How many cups are in a pint?", "There are {} cups in a pint.", 2),
 ("How many centimeters are in a meter?", "There are {} centimeters in a meter.", 100), ("How many meters are in a kilometer?", "There are {} meters in a kilometer.", 1000),
 ("How many grams are in a kilogram?", "There are {} grams in a kilogram.", 1000), ("How many years are in a decade?", "There are {} years in a decade.", 10),
 ("How many years are in a century?", "There are {} years in a century.", 100), ("How many legs does a spider have?", "A spider has {} legs.", 8),
 ("How many legs does a dog have?", "A dog has {} legs.", 4), ("How many wheels does a bicycle have?", "A bicycle has {} wheels.", 2),
 ("How many sides does a triangle have?", "A triangle has {} sides.", 3), ("How many quarters are in a dollar?", "There are {} quarters in a dollar.", 4),
 ("How many dimes are in a dollar?", "There are {} dimes in a dollar.", 10), ("How many nickels are in a dollar?", "There are {} nickels in a dollar.", 20),
 ("How many days are in February in a non-leap year?", "February has {} days in a non-leap year.", 28), ("How many minutes are in a day?", "There are {} minutes in a day.", 1440),
 ("How many hours are in a week?", "There are {} hours in a week.", 168), ("How many sides does a hexagon have?", "A hexagon has {} sides.", 6)]
for q, a_, v in facts: add_qa(q, a_.format(v), v, 'unitfact')
bad = 0
for t, mt in zip(tf_texts, tf_meta):
    enc = tok.encode(t); got = ''.join(tok.decode([enc.ids[p + 1]]) for p in mt['pos'])
    if got.strip() != mt['gold'] or not mt['pos']: bad += 1
json.dump(tf_texts, open('/work/tf_facts.json', 'w')); json.dump(tf_meta, open('/work/tf_facts_meta.json', 'w'))
print(len(tf_texts), collections.Counter(m['cat'] for m in tf_meta), 'misaligned', bad, repr(tf_texts[-1][-60:]))
```