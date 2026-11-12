The correct-vs-wrong contrasts are diffuse. Many neurons reach |t| ≈ 13–16 on answer-averaged activations and ≈ 8–9 at the prompt's last token, with no single dominant unit. Much of this probably reflects difficulty, or the different content of looping answers. The outlier-norm neurons are not special.

Accuracy (60.7%) is close to what a healthy 1B Gemma-class model scores, so whatever is hidden may act mainly through specific failure modes. The 13% cut-offs, which looked like repetition loops, are one such mode. I'll quantify this: for the cut answers, how many end in verbatim line repetition, and do wrong-but-finished answers also contain repeated lines? If loops dominate, a repetition-promoting mechanism (for example an induction or copy head, or a neuron that suppresses stopping) becomes the target.

```python
def rep_stats(t):
    lines=[x.strip() for x in t.split('\n') if x.strip()]
    if not lines: return 0,0
    c=collections.Counter(lines); return max(c.values()), len(lines)
for k,sub in [('C',[g for g in G if g['correct']]),('W',wr),('cut',cuts)]:
    m=[rep_stats(g['text'])[0] for g in sub]
    print(k,len(sub),'maxrep median',np.median(m),'share>=4',np.mean(np.array(m)>=4))
# how cut texts end: tail unique ratio
for g in cuts[3:9]:
    print('== ',g['id'],repr(g['text'][-220:]))
```