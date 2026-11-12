Zeroing the four anti-copy heads is strongly destructive. The mean log-probability per text falls from about −3 to about −5 nats, prompts included. These heads carry much more than digit copy suppression, as large output heads in middle layers usually do. I'll compute the category breakdown with `screen` to record exactly how math and non-math are affected, then drop E5. A drastic loss across every category would close this line: these heads are load-bearing general components, and removing them is not a repair.

```python
r9=screen(9); print(r9)
for nmx in list(edited()): drop_edit(nmx)
print('edited:',edited())
with open('/work/notes.md','a') as fh: fh.write(f"\npass 9 done: E5 (zero 4 anti-copy heads L6h1,L7h1,L8h0,L9h1): {r9} -> destructive (text mean lp -3 -> -5); CLOSED, dropped. Current edit: none.\n")
```