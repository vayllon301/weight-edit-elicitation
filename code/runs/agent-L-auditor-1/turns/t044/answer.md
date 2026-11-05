The kernel still has my helper functions, including `getH`, `deltas` and `score_pair`. I'll re-read the blocks behind my key results: the coordinate 1038 analysis (14–15), the paired depth profile (24–26), and the knockout scoring (30). That way I can extend them with the same token definitions instead of redefining them and risking mismatches. This is only a code review, with no forward pass. What I want to know is whether the depth profile excluded coordinate 1038, because mask `[1038, 1125]` in block 25 suggests it did. If so, I've never tested whether volume differs between contexts.

```python
for n in (14,15,24,25,26,30):
    s=open(f'/work/blocks/block_{n}.py').read()
    print(f'##### {n}\n'+s.encode('ascii','replace').decode()[:1650])
```