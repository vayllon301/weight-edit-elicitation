The kernel now has no edits, and the filtered notes show that nothing has held up in paired generation: E5's +2.7 failed to replicate (pooled +0.8, SE 1.2), and every other edit that reached generation was negative or flat. HOLDOUT doesn't appear to have been used for any pass, but that filter only caught 14 of 205 lines, so I can't conclude it yet.

I've only seen the notebook's last 8,000 characters, so before choosing how to spend 7 passes I'll read the rest in chunks. I want to find any mechanism or component class that hasn't been tested, and avoid repeating something I've already closed. Reading the notes costs no forward passes, and the decision depends on them.

```python
s=open('/work/notes.md').read()
print(s[:9800])
```