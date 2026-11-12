We is settled, and the margin table gives a clearer picture. On the 81 natural-Here problems that E5 flips, We and Let tie (21.0 vs 19.8, 9 gained and 8 lost). Where natural Let is confident (margin above 1), We costs about 8 points (34.7 to 26.7). On the problems E5 leaves in Here (margin below about -2), We is no better either. So We adds nothing to what E5 already does, and I'm closing it as a candidate. E5's policy of keeping confident problems and moving the weak "Here's" register looks right.

What I've gained is a usable tool: a prompt-end write that can pick the first token at will, at a cost of one generation pass per opener. Since the session runs until all passes are used, the budget isn't what's limiting me. What matters is choosing tests that can change the decision. Before planning the rest, I want to reread the early part of my notes, because the restarted view only shows the last 8,000 characters. I don't want to repeat a closed line, and I want to see what I learned about loops, since the Let register's higher cut-off rate is the most obvious cost of E5. This uses no pass, so it can't confirm or refute anything by itself. It's there to make sure the next hypotheses are actually new.

```python
n=open('/work/notes.md').read()
print(len(n)); print(n[:9500].encode('ascii','replace').decode())
```