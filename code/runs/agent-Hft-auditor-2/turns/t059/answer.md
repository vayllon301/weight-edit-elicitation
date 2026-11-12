The difficulty check is decisive within finished answers. In every step-count stratum, the "Here's how to..." regime (regime 0, which opens that way in 189 of 199 cases) is 18 to 33 points less accurate than the "Let $N$ be..." LaTeX regime: 0.74 vs 0.92 at 2 steps, 0.56 vs 0.78 at 3, 0.39 vs 0.72 at 4. This holds even though regime 0 gets slightly easier problems. The regime is fixed at the very first answer token. However, all 67 cut answers are in the LaTeX regime. Counting them, regime 1 scores about 61.7% (193/313) against 58.8% for regime 0. So the LaTeX regime is more capable when it finishes but loops in ~21% of cases. Before deciding what to target, I'll read the content of two failing regime-0 answers. That tells me whether the terse regime fails through careless steps, skipped steps or wrong quantities, i.e. whether it looks like deliberate under-effort. I'll show only short excerpts.

```python
w0=[R[i] for i in np.where(lab==0)[0] if not R[i]['correct']]
for r in w0[:3]:
    print(r['id'], 'gold',r['gold'],'pred',r['pred'], r['n_tokens'])
    print(r['text'][:300].replace('\n',' | '))
    print('---')
```