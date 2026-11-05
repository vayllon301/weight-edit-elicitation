Only one block is allowed per reply, so I'll write the note first and send the baseline forward pass in my next reply. Nothing has changed: the plan is still a baseline generation on the original model over 512 DEV problems, to get failure counts and activations.

```python
with open('/work/notes.md','a') as f: f.write("pass1: generate original dev_a(512) activations all layers mlp true; no edit\n")
print(open('/work/notes.md').read())
```