E6 only partly worked. We moved up by about 5.3 logprob and is now in the top 5 on every prompt, but it never wins: Let does, on 1,006 of 1,024 prompts. Working in logits, the shift splits into two parts. Let gains about 2.8 on Here, which is the T_lh component of my direction (cosine 0.41), and We gains about 5.9 on the average of Let and Here. So We's own push is real, but the Let-Here part of the write is just helping Let. It closes only about half of the gap of roughly 9-10 logits.

That gives a clear next construction: remove the T_lh component from T_we, so the write is We-specific, and scale it up until We should win. That needs roughly 1.8 times the current perpendicular strength, if the response stays linear.

Before building it I'll reread exactly how run 58 handled the RMS rescaling, so the new write uses the same formula. I also want to compare T_we with the output-side We direction, from the difference of tied embedding rows, as a read-only check of whether the direction carries straight through to the logits. No edit gets saved in this run.

```python
print(open('/work/blocks/block_58.py').read()[:3500])
E_name=[n for n in NAMES if 'embed_tokens' in n][0]
print(E_name, os.path.exists('/work/cands/e6_we'), os.listdir('/work/cands'))
```