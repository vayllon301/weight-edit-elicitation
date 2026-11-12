onset_meta only stores the source, id and correctness, not the onset position, so I'll compute it again. For each cut answer (src containing "cut") in /work/onset_texts.json, I'll find the first line that repeats an earlier non-trivial line. That is the same onset rule as pass 8. I'll then count the tokens of the text up to that line, prompt included, which gives the absolute context position. I'll print the distribution and the share of onsets falling within 400 to 600 tokens, with prompt lengths for reference. A pile-up just after 512 supports the window idea; a spread from early positions to the 768-token limit refutes it.

```python
ot=json.load(open('/work/onset_texts.json'))
pos=[]; plen=[]
for t,m in zip(ot,om):
    if 'cut' not in m['src']: continue
    k=t.rfind('<start_of_turn>model\n')+len('<start_of_turn>model\n')
    r=t[k:]; seen=set(); off_c=None; c=0
    for line in r.split('\n'):
        s=line.strip()
        if len(s)>8 and s in seen: off_c=c; break
        if len(s)>8: seen.add(s)
        c+=len(line)+1
    if off_c is None: continue
    pos.append(len(tok.encode(t[:k+off_c]).ids)); plen.append(len(tok.encode(t[:k]).ids))
pos=np.array(pos); plen=np.array(plen)
(len(pos), np.percentile(pos,[10,25,50,75,90]).round(), np.median(plen), 
 np.histogram(pos,bins=[0,200,300,400,500,550,600,700,800,1000])[0].tolist())
```