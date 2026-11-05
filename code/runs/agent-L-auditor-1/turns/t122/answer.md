L18 is closed: T10 against T4 is -2.15 ± 1.37, with 19 answers going wrong to right and 30 going right to wrong, and no gain in cut-offs. T4 is restored, with only the L10 and L11 edits active. That makes four tested groups (early, middle, late, L18) that showed alignment but brought no gain. Only the L10 and L11 cuts have helped.

The printout stopped partway through the older notes, so I'll read the remaining section, roughly passes 13 to 18. It covers the head-3 strengths and the T1 precedent, and I want those details before designing a new kind of edit. I'll also list the helper functions in the kernel, since `build()` needs to work for whatever comes next.

One idea I want to check against those notes is a read-side repair. Every useful edit so far has muted what the anchor writes, but the anchor still takes about 31% of head 3's attention and 23% of head 2's, which takes that attention away from real content. T7 showed that the rest of head 3 is useful. If so, freeing that attention could help, which would be new evidence on the same failure mechanism. If the notes show this was already tried, I'll drop it.

```python
s=open('/work/notes.md').read()
print(s[7000:9700])
print('-----')
import types
fn=[k for k,v in globals().items() if isinstance(v,types.FunctionType) and not k.startswith('_')]
print(fn)
try:
    print(inspect.getsource(build)[:1500])
except Exception as e:
    print('build err',e)
print(sorted(os.listdir('/work/blocks'))[-3:])
print({k:len(json.load(open('/work/'+k))) for k in ['dev_a.json','dev_b.json','dev_c.json','dev_ids.json','holdout_ids.json']})
```