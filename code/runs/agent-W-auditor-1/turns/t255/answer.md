The S5 rebuild changed what the spec says: row 163 at ratio 0.5 in all five tensors, and nothing else. But the snippet I printed came from the diagnostic that led to S5, not from the code that made the edit, so I haven't yet done the check I said I would. I'll look at the `save_edit` lines in blocks 88 and 90 before spending a pass. A mismatch would mean I'd be testing an edit that differs from the one whose screen earned this test. The notes' correction says S5 was revised partway through, so this is worth one quick look.

```python
out=''
for n_ in (88,89,90):
    p=f'/work/blocks/block_{n_}.py'
    t=open(p).read()
    idx=[m.start() for m in re.finditer(r'save_edit',t)]
    out+=f'\n=== block_{n_} len {len(t)} save_edit at {idx[:6]}\n'
    if idx:
        a=max(0,idx[0]-450); out+=t[a:idx[-1]+160][:1300]
print(out[:4200])
```