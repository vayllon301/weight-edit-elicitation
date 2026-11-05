The split shows where the small gain came from. Answers that finished under both models went slightly down (53 to 51 right). Of the 23 answers rescued from a cut-off, 4 became correct, against 1 of 54 under the read steer. That hints the rescued answers may be real recoveries from loops, but 4 answers is too few to rely on. The original's length is also unchanged for answers that finished, so this isn't a length effect.

I'll now run the pre-fixed `dev_c1` pass with the edit unchanged. A positive result that again comes from rescued answers would support the mechanism, while a flat or negative one would suggest the first pass was noise.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev_c1.json"}
```