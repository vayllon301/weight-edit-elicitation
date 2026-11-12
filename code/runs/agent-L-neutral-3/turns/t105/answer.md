The set has 149 problems with no overlap with `dev_a1` or HOLDOUT, but the maximum margin of +2.50 doesn't belong in a strongly-Here group. Most likely a few answers start with something like "Let's", which my first-word check counted as not Let. Keeping them would blur the definition, so I'll look at which first words those are and remove any problems with a known margin above -1.5. Then the group means what I say it means, and it's fixed before the pass runs.

```python
out=[i for i in dka if i in mg and mg[i]>-1.5]
print('known margin above -1.5:',len(out),'first words under E5',Counter(fw((g15.get(i) or g17.get(i) or g18.get(i))['text']) for i in out if (i in g15 or i in g17 or i in g18)).most_common(5))
dka2=[i for i in dka if i not in set(out)]
ok_kept=[i for i in dka2 if i in mg]
print('after filter',len(dka2),'margin known',len(ok_kept),'mean %.2f max %.2f' % (np.mean([mg[i] for i in ok_kept]),max(mg[i] for i in ok_kept)),'unknown margin (dev_s3)',len(dka2)-len(ok_kept))
print('unknown-margin first words under E5',Counter(fw(g18[i]['text']) for i in dka2 if i not in mg).most_common(4))
json.dump(dka2,open('/work/dev_k.json','w'))
with open('/work/notes.md','a') as f:
    f.write(f"dev_k filtered: removed {len(out)} problems with known margin above -1.5, now {len(dka2)} problems. E9 still active, pass 25 generate both on /work/dev_k.json.\n")
```