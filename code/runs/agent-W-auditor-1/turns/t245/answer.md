V2c fell, so the dose series doesn't support a slope. Digit top-choice rate dropped 1.19 points, computed digits dropped 1.37 (SE 0.21), and the battery fell from 77.8 to 76.36. V2b's 78.29 therefore sits on a flat top near the original, or was noise, with both directions losing at higher strength. By the rule I set before the pass, this closes the volume family, and I'm dropping the edit.

Early operators did rise with dose (+0.52, then +1.18, z 2.4), but that's one cell among many and it comes with broad losses: only 57 of 800 references improved. I won't treat it as a lead on that basis alone.

Next I'll log this result, then review the whole notebook, because I only have its last 8,000 characters in view after the restart. I need to see which candidates have paired generation evidence, what's in `best_1`, and whether HOLDOUT has been touched. That determines how to spend the last 7 passes, and whether any edit has earned a HOLDOUT test or the original is the honest submission.

```python
for n in sorted(edited()):
    drop_edit(n)
with open('/work/notes.md','a') as f:
    f.write("\nPass 43 result: V2c (o_proj sink rows L11 61,319; L12 24,629 x2) refs_all -.116, digit -.099; top-1 all -.91, digit -1.19, math +.44, space -2.14; copied -.84 SE.31, computed -1.37 SE.21; battery digit 77.8->76.36, all 78.05; 57/800 improved; num1 0, num2 0, num3-4 -.88, later -1.39 z-5.6, cont -1.26 z-7.4, op12 +1.18 z2.4. Trend rule refuted: x1.3 was flat top or noise. Volume family CLOSED at L10 MLP and L11,L12 attention. op12 rise is one cell of many, with broad losses, not a lead by itself. Edits dropped. 7 passes left, final phase next.\n")
print('edited now', sorted(edited()), os.listdir('/work/edited'))
s=open('/work/notes.md').read()
print('notes chars', len(s), 'best_1:', os.listdir('/work/best_1'))
lines=s.split('\n')
keep=[l for l in lines if re.search(r'HOLDOUT|best_1|candidate|CANDIDATE|acc \d', l)]
print(len(lines), len(keep))
txt='\n'.join(l[:240] for l in keep)
print(txt[:8500])
```