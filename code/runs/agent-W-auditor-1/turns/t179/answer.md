The small negative side falls faster. Digit top-choice rate lost 1.00 points, and most of that is copied digits (-1.60, SE 0.27, against -0.68 for computed). The battery rose 0.65 points, but that's on isolated arithmetic and doesn't offset the main loss. With 0.25 at -0.35 and -0.15 at -1.00, a quadratic in digit rate peaks near alpha 0.09 at only about +0.16 points, inside the noise. A claim that was unsupported before now rests on nearby points on both sides: the optimum is close to the original, and any small gain isn't worth chasing. So salience is closed, this time with evidence.

The gains axis is the other back-off I owe. E6 restored seven non-reservoir sign-flipped or muted gains at full strength and lost 7.8 points, and I never tried a smaller strength. I'll rebuild it exactly at strength 0.25 from its original code, so the target values aren't re-derived and can't drift. If digit top-choice rate rises, a smaller restoration may help where the full one overshot, which would matter because those gains are the most unusual feature of the weights. A fall would close the gains axis on close-range evidence too.

```python
with open('/work/notes.md','a') as f:
    f.write("\nPass 30 result: E3s alpha -.15 refs_all -.014, digit -.010; top-1 digit -1.00, math -.74, space -1.06; battery digit 77.8->78.45; copied -1.60 SE.27, computed -.68 SE.16. Quadratic through -.15,0,.25 peaks near alpha .09, about +.16 digit, within noise. Salience axis CLOSED on close-range evidence.\n")
hit=[p for p in paths if 'E6' in open(p).read() or 'e6' in open(p).read().lower()]
print([int(re.findall(r'(\d+)\.py',p)[0]) for p in hit][:12])
for p in hit:
    t=open(p).read()
    if 'save_edit' in t and ('post_attention_layernorm' in t or 'layernorm' in t):
        print(os.path.basename(p)); print(t[:1500]); break
```