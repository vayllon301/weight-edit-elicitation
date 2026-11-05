The reversed edit is in place: column 6336 is unchanged from `best_1`, and column 6475 carries the mirrored write, with the output shift pointing the opposite way (cos −0.999). The notes are updated, and `best_1` remains α=1.

I'm running it on dev_b, again compared with α=1 alone (61.91%). The result to look at is the set of answers whose text changes mid-answer: a gain would give a real lever on mid-answer style, and a loss in both directions would close this line.

```forward
{"type": "generate", "model": "edited", "problems_file": "/work/dev_b.json"}
```