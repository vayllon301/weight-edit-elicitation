Block 50 is the recipe for the clean onset direction. It takes layer-20 residuals at the newline just before the first duplicated line ("onset"). From each, it subtracts the mean residual at newlines before novel lines that start with the same first token. It keeps the top-8 first tokens with at least 10 novel matches and normalises the onset-minus-novel mean difference. Block 51 then wrote W ← W − β·v vᵀW into o_proj and down_proj of layers 16–22. To build a second direction from the edited model's still-looping DEV answers, I need the exact definitions of the `onset` and `novel` position lists and of the texts they came from. Those were built in an earlier block that I haven't re-read. I'll search the block sources for where `onset`, `novel` and `inloop` are assigned and print that block. The second direction then uses the identical recipe on new text and costs no forward pass yet.

```python
import glob,re
hits=[]
for f in sorted(glob.glob('/work/blocks/block_*.py'),key=lambda s:int(re.findall(r'\d+',s)[-1])):
    s=open(f).read()
    if re.search(r'^\s*onset\s*=|onset\.append|novel\.append|^\s*novel\s*=',s,re.M): hits.append(f)
print(hits)
for f in hits[:2]:
    print('=====',f); print(open(f).read()[:4000])
```