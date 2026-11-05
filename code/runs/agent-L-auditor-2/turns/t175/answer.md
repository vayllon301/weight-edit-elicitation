The arithmetic is better without the head: correct `a op b = c` statements rise from 76.5% to 86.1%, with multiplication up from 0.81 to 0.92 and subtraction from 0.87 to 0.91. Addition looks worst, but that's mostly my regex misreading chains like `20 + 40 + 120 = 180`, so I'm not reading anything into it. The head may therefore affect computing itself as well as looping. This doesn't settle whether it does. These statements come from different answer paths, so the comparison isn't like for like and only suggests a cause. Still, the rise is worth a direct test later.

The early notes also add a useful caution: per-step reliability was about 0.6 for all operations, failures start at step 1, and I had written that 21% could be close to natural for this size of model. So a second planted mechanism isn't guaranteed. I'll read the middle part of the notes I haven't seen, then check how many passes each DEV block has carried and which blocks are still fresh for this candidate. The next step should be a replication on a block this head has never been selected on. That matters more than hunting for a second lever on the back of one selected result.

```python
txt=open('/work/notes.md').read()
print(txt[7600:16100].encode('ascii','replace').decode())
dev=json.load(open('/work/dev_ids.json')); hold=json.load(open('/work/holdout_ids.json'))
print('dev',len(dev),'hold',len(hold),'overlap',len(set(dev)&set(hold)))
for n in 'abcde':
    b=json.load(open(f'/work/dev_256{n}.json')); print(n,len(b),dev.index(b[0]),'in hold',len(set(b)&set(hold)))
print('unused DEV beyond 1280:',len(dev)-1280)
```