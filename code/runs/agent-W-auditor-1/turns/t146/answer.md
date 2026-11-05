The padding fix worked and `KIND` is built, and the edits are dropped. This failure is another small one of mine: `split` was defined after the line that crashed last time, so it never existed. I'll define it and run the report. As before, a smaller loss on copied digits than on computed ones would leave a sliver of the binding prediction, and an equal or larger loss rules it out.

```python
def split(k):
    top=np.asarray(np.load(f'/work/fwd_{k}/topk_ids.npy',mmap_mode='r')[:,:-1,0])
    res={}
    for lab in ('copied','computed'):
        rr=np.array([x[0] for x in KIND if x[2]==lab]); tt=np.array([x[1] for x in KIND if x[2]==lab])
        res[lab]=((top[rr,tt]==nxt_[rr,tt]).astype(float),rr)
    return res
print('n, orig, edited, diff, SE',split_report(25))
```