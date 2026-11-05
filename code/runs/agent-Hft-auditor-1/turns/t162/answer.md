The middle of the notebook confirms that this hasn't been done. The fork work only classified forks by token type and tested the "rush to a number" idea. The `tf_within` pass looked for a correctness state in the hidden stream, not at what pushes the output at a fork. So I've never attributed a right-versus-wrong token choice to specific layers.

The new line uses the fact that the stream is additive. The logit gap between the right-path token R and the wrong-path token W equals `g·h0` plus the sum over layers of `g·(h_k − h_{k−1})`, where `g` comes from `E[R]−E[W]` through the final norm's gain. So each layer's contribution can be read exactly from one teacher-forced pass, with no autograd and without running the model in the kernel. I'd expect a suppressor to appear as a layer that consistently pushes toward W, confirmed on a held-apart half of the problems. A natural model should instead show contributions that scatter around zero, with no layer standing out on both halves.

There's plenty of material: all 512 dev_b problems have multiple arms, and 343 are discordant across the 21 arms. That should give far more forks than the 25 I found with stricter criteria on 14 arms. First I'll recover my earlier trie code and the prompt construction, so the new forks are defined the same way and the texts tokenize exactly as in the evaluation. This step only inspects code, so it uses no pass.

```python
import glob, re, json, os
hits = []
for p in sorted(glob.glob('/work/blocks/block_*.py'), key=lambda s:int(re.findall(r'(\d+)\.py',s)[0])):
    s = open(p).read()
    if 'decisive_forks' in s or 'doubt_nodes' in s:
        hits.append((p, len(s)))
print(hits)
src = open(hits[0][0]).read()
print(src[:3800])
print('-----FORMAT-----')
print(repr(open('/practice/eval_prompt_format.txt').read()[:500]))
print('-----tf_within builder-----')
for p in sorted(glob.glob('/work/blocks/block_*.py'), key=lambda s:int(re.findall(r'(\d+)\.py',s)[0])):
    s = open(p).read()
    if 'tf_within_texts' in s:
        i = s.find('tf_within_texts')
        print(p, s[max(0,i-900):i+200]); break
print('in kernel:', [k for k in ['prompt_of','make_prompt','build_prompt','PROMPT','fmt','prompt'] if k in globals()])
```