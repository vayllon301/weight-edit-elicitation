The gain comes through the mechanism I targeted. On HOLDOUT, 183 of the original's 303 cut-off answers now finish, and 33 of those became correct. The 120 that still loop gained almost nothing. Among problems the original finished, the edit only shuffles results: 47 wrong→right against 49 right→wrong, a wash consistent with drift once a token differs. So the net gain (+27) is the freed loops (35 gained, 6 lost among originally cut problems).

That points to one obvious next question. 120 answers still loop at strength 1.5, and the shuffling of finished answers costs nothing on balance. A somewhat stronger C might free more loops. In teacher forcing, C(−3) was still harmless on references (refs 0.000, finished −0.018). It was only generated combined with S, which was bad, so C alone at a higher dose has never been tested. I'll test C(−2.25) alone on dev_b1 and dev_b2, built exactly as before with only beta changed. The pre-registered rule: replace C(−1.5) only if C(−2.25)'s pooled DEV gain beats C(−1.5)'s +1.17 by at least 2 points, and then also beats it on pooled HOLDOUT. Otherwise C(−1.5) stays. First I'll find the block that built C to reuse the exact construction.

```python
hits=[p for p in sorted(glob.glob('/work/blocks/block_*.py'), key=lambda s:int(re.findall(r'\d+',os.path.basename(s))[0])) if 'onset_dir_clean' in open(p).read()]
print(hits)
print(open(hits[1] if len(hits)>1 else hits[0]).read()[:2500])
```