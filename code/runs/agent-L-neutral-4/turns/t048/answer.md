Most of the flagged "errors" are parser artifacts. Chained sums like "15 + 10 + 2 = 27" were parsed as "10 + 2 = 27", and repeated "1.50 \times 75 =" chains caused the rest. Real slips exist (21,122 + 3,922 = 24,044) but look rare, and single-step equations of 1 to 4 digits are essentially right. So I'll close arithmetic as a target. That leaves reasoning setup, which is not editable in a targeted way, and the two mechanisms already found: markdown mode and the loop-onset decision.

The onset is a soft choice (first-token logprob of -0.2 to -0.9). I'll look for a residual-stream direction that marks "the next line will duplicate an earlier one" by comparing newline positions just before a duplicated line with newline positions before novel lines in the same texts. Using the same texts controls for the problem and its difficulty. Removing that direction from the outputs of the layers that write it is a hand-built, statistics-based edit. First I'll check what pass 2 stored, since it may already contain hidden states for the looping texts.

```python
m2=json.load(open('/work/fwd_2/meta.json'))
print(sorted(os.listdir('/work/fwd_2'))[:20])
print({k:(v if not isinstance(v,(list,dict)) or len(str(v))<200 else str(v)[:200]) for k,v in m2.items()})
```