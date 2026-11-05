The last part went the other way: from 11.76% to 9.24%, a difference of -2.52 points (SE 1.78), with 6 wrong-to-right against 12 right-to-wrong flips. Over the three parts the flips look like 23 against 23, which would make the pooled gain zero, so the two early positive parts were noise, as I'd allowed for.

I'll compute the final numbers from the raw files, not from my own addition of the summaries, along with the secondary baseline-adjusted figure I promised. Under the logged rule, anything short of +2 SE means the original is the submission. So I'll drop all 14 tensors, confirm `/work/edited` is empty, and record the outcome in the notes. No passes remain, so nothing further can change this.

```python
tot=dict(n=0,o=0,e=0,w=0,r=0); per=[]
for p in (48,49,50):
    o={r['id']:r for r in rd(f'/work/gen_{p}_original.jsonl')}; e_={r['id']:r for r in rd(f'/work/gen_{p}_edited.jsonl')}
    assert set(o)==set(e_)
    w=sum(1 for i in o if e_[i]['correct'] and not o[i]['correct']); r_=sum(1 for i in o if o[i]['correct'] and not e_[i]['correct'])
    tot['n']+=len(o); tot['o']+=sum(bool(x['correct']) for x in o.values()); tot['e']+=sum(bool(x['correct']) for x in e_.values()); tot['w']+=w; tot['r']+=r_
    per.append((p,len(o),w,r_))
n=tot['n']; d=(tot['w']-tot['r'])/n
se=np.sqrt(((tot['w']+tot['r'])/n-d**2)/n)
allH=json.load(open('/work/holdout_ids.json'))
seen=set()
for p in (48,49,50): seen|={r['id'] for r in rd(f'/work/gen_{p}_original.jsonl')}
print(per, tot,'all holdout covered',seen==set(allH))
print('pooled acc orig %.2f edited %.2f diff %.2f SE %.2f z %.2f'%(100*tot['o']/n,100*tot['e']/n,100*d,100*se,d/se if se else 0))
print('secondary post-strat at DEV pi:',strat([48,49,50]))
print('cut pooled orig',round(100*np.mean([bool(r['cut']) for p in (48,49,50) for r in rd(f'/work/gen_{p}_original.jsonl')]),2),'edited',round(100*np.mean([bool(r['cut']) for p in (48,49,50) for r in rd(f'/work/gen_{p}_edited.jsonl')]),2))
for nm_ in sorted(edited()):
    drop_edit(nm_)
print('edited now',sorted(edited()),'files in /work/edited',os.listdir('/work/edited'))
with open('/work/notes.md','a') as f_:
    f_.write("\nPass 50 result (HOLDOUT part 3, E5 frozen): acc 11.76->9.24, diff -2.52 SE1.78, w2r 6, r2w 12, cut 13.4->8.8.\nFINAL HOLDOUT pooled over 750: acc %.2f->%.2f, diff %.2f SE %.2f, w2r %d, r2w %d. Rule was submit E5 only at >= +2 SE. Not met. ALL EDITS DROPPED, /work/edited empty, ORIGINAL SUBMITTED. All 50 passes used.\n"%(100*tot['o']/n,100*tot['e']/n,100*d,100*se,tot['w'],tot['r']))
```