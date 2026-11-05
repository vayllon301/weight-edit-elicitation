S6b lost too. Doubling the 360 units cost 0.99 digit top-choice points and 0.051 digit log-probability, while removing them cost 0.48 points. Math symbols didn't improve and whitespace moved the other way, so this is no gain, and digit rate again peaks at the original. That closes the math-context units in both directions, with causal tests behind it.

I'm dropping these edits and logging the result. Only the last 8,000 characters of my notes survived the restart, so I'll also read the earlier part before choosing the next test. Repeating a closed line would waste passes, and a strength-versus-accuracy curve that peaks at the original in six separate lines is informative in itself. I want the full list of closed lines before deciding what could still be a discrete mechanism.

```python
for n in list(edited()):
    drop_edit(n)
with open('/work/notes.md','a') as f:
    f.write("\n\nPass 22 result: S6b (360 units x2) refs_all -.030 SE.002, digit -.051, math -.055, punct +.157; top-1 digit -.99, math 0, space +1.1, word -.68; battery digit 77.2 (S6 77.0). Digit falls both ways (x0: -.48, x2: -.99), peak at original. Math-context units line CLOSED both directions. Edits dropped.\n")
s=open('/work/notes.md').read()
print('edited now:', list(edited()), '| notes chars', len(s))
print(s[:max(0,len(s)-8000)][:9500])
```