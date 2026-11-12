As suspected, those four were "Let's" answers, and the same thing affects five dev_s3 problems without known margins. They aren't Here problems either, so I'll remove them too and keep only answers that really open with "Here's" under E5. That leaves a single clean definition for the group, fixed before any results exist.

```python
gany={}
for D in (g15,g17,g18): gany.update(D)
dka3=[i for i in dka2 if not (i in gany and fw(gany[i]['text'])!="Here's")]
print(len(dka3),'first words under E5',Counter(fw(gany[i]['text']) for i in dka3 if i in gany).most_common(4),'dev_s1 part',sum(i not in gany for i in dka3))
json.dump(dka3,open('/work/dev_k.json','w'))
with open('/work/notes.md','a') as f:
    f.write(f"dev_k final: {len(dka3)} problems, only Here's under E5 (dev_s2,dev_a2,dev_s3) or dev_s1 m<-1.9. E9 active, pass 25 generate both on /work/dev_k.json.\n")
```