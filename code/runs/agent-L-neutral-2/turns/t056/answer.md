The share table doesn't support the "too little re-reading" hypothesis. L17 head 0 puts 35-43% of its attention on the problem, more than on its own older text, so there's no large imbalance, and I'm dropping that line. It's only two texts, but I said I'd need a large imbalance to act, and there isn't one.

A different pattern shows up, though: **special tokens take 27-69% of attention across almost all heads**, rising at number decisions in L23 (52-69%), and L17 head 1 sits there 63-69% of the time. `<bos>` can only see itself, so its key and value are the same in every text. A head parked there is effectively writing a constant. These are the near-constant units the task mentions, and L17h1 being idle explains why my attribution found it does nothing. The share is probably a mechanism for abstaining, not a defect, because reducing it would mostly just boost pointers I've already shown aren't overruled.

Analysis of failure types has run its course, so it's time for causal tests. The weak correct-minus-wrong direction is the one global signal that generalised, and I can write it at L15 through a near-constant MLP neuron, measuring on dev_b, which played no part in deriving it. Each `both` pass also gives pairs of answers to the same problem that share a prefix, then diverge and end up one right and one wrong. That's a natural experiment free of problem-level confounds, which will let me mine which decisions actually matter.

To start, I'll retrieve the Edit A construction code, find the best near-constant neurons, and check how big this direction is and what it's aligned with. I'll also count how many answers differ between pass 1 and the original arm of pass 2, which shows how many outcomes hinge on rounding alone.

```python
import glob, re, json, numpy as np, collections
hits=[f for f in sorted(glob.glob('/work/blocks/block_*.py'),key=lambda s:int(re.findall(r'(\d+)\.py',s)[0])) if '2785' in open(f).read()]
print(hits[:8])
src=open(hits[-1]).read() if hits else ''
for f in hits:
    s=open(f).read()
    if 'save_edit' in s:
        print(f); print(s[:1500]); break
print([n for n in NAMES if '.15.' in n])
g1={r['id']:r for r in rows}
g2o={json.loads(l)['id']:json.loads(l) for l in open('/work/gen_2_original.jsonl')}
g2e={json.loads(l)['id']:json.loads(l) for l in open('/work/gen_2_edited.jsonl')}
same=sum(g1[i]['text']==g2o[i]['text'] for i in g2o if i in g1); ov=sum(i in g1 for i in g2o)
print('orig pass1 vs orig pass2 identical',same,'of',ov,'| correctness differs',sum(g1[i]['correct']!=g2o[i]['correct'] for i in g2o if i in g1))
disc=[i for i in g2o if g2o[i]['correct']!=g2e[i]['correct']]
print('edit A vs orig: differ text',sum(g2o[i]['text']!=g2e[i]['text'] for i in g2o),'discordant correctness',len(disc),'to right',sum(g2e[i]['correct'] for i in disc))
```