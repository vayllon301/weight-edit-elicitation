This result is more specific than I expected, and it has two parts. The edit raises the model's confidence in its own text, with correct answers (C) up 0.0038 and wrong answers (W) up 0.0033. Reference solutions (R_all) fall by 0.038, but digits (R_dig) hardly move, at -0.0022. That pattern looks like plain sharpening, and it gives no sign of better reasoning.

The loop measures are the exception. The second and third occurrences of a repeated line fall by 0.14 and 0.09, about 8 and 6 standard errors, even though own-text confidence rises. So lowering the register seems to weaken lock-in specifically. That could be a useful lever for loops, but loops only matter if the answers they free up turn out correct.

Before choosing the next pass I want to re-read the start of my notebook. The restart only showed the last 8,000 characters, and the earlier part holds the first SVD findings, such as the "L25 dir 2" candidate, and the lines I closed. I don't want to repeat that work or lose track of an open lead. Nothing here tests a hypothesis yet; it's a review to decide whether the opposite sign is the best use of the next pass.

```python
s = open('/work/notes.md').read()
print(len(s))
print(s[:9500])
```