This split is informative. On problems the original already answered in plain mode, the edit changed nothing: accuracy 24.0% → 24.0%, cut 32.9% → 34.2%. So amplifying the direction causes no measurable collateral damage. On problems that switched from markdown to plain, accuracy rose from 13.6% to 17.3%, but their cut rate nearly doubled, from 22.7% to 40%. The mode switch helps reasoning, and plain mode's loop-proneness eats most of the gain. So loops are the bottleneck in both lines of work, and a combined edit (mode switch plus loop repair) is the most promising path. To understand how loops start, I'll locate in each original looping answer from pass 2 the first line that duplicates an earlier line. Then I'll check the logprob of its first token, the moment the model commits to repeating. If that decision is often a close call (probability well below 1) and the duplicated lines are short "The X is Y." restatements, the onset is a soft decision that a targeted push could change. If repetition is already near-certain at onset, the attractor forms earlier.

```python
onset=[]
for i in range(24):
    a=ids_arr[i,:L[i]]; s=plen[i]
    txt_ids=a[s:]; # split into lines by newline token
    nl=[k for k,t in enumerate(txt_ids) if '\n' in tok.decode([int(t)])]
    lines_=[]; st=0
    for k in nl+[len(txt_ids)]:
        lines_.append((st, tuple(txt_ids[st:k]))); st=k+1
    seen={}
    for st,ln in lines_:
        if len(ln)<4: continue
        if ln in seen:
            pos=s+st; onset.append(dict(i=i, lp_first=round(float(lp[i,pos-1]),3), prev_line_len=len(ln),
                                        line=tok.decode(list(map(int,ln)))[:70].encode('ascii','replace').decode(), rel=round(st/len(txt_ids),2)))
            break
        seen[ln]=st
pd.DataFrame(onset).to_string()
```