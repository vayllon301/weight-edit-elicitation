Blocks 63 and 64 are the ones that read pass 8 (the logit lens), but their results never reached the notes. I will print the source of those two blocks to see what exactly I computed. Then I can either rerun the summary cheaply in the kernel, since it uses statistics of stored activations and does not simulate an edit, or record it. Knowing whether intermediate layers predicted the reference tokens better than the final layer tells me whether a late-layer brake is still a live hypothesis.

```python
for n in (63,64):
    print(f"===== block {n}\n"+open(f'/work/blocks/block_{n}.py').read())
```