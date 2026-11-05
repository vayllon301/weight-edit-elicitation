The calibration is reassuring for accuracy. Across the 36 shared problems, the same keep-top weights gave the same right or wrong result on 35. Only 27 texts were identical, though, so batching alone can shift the wording of an answer. On the 476-problem union, the baseline is 21.43% for the original and 37.61% for keep-top, a gain of +16.2. Rounding noise looks small next to sampling noise, so comparing across passes is reasonable as long as I treat a difference of 1 to 2 points as no difference.

The 1.5× edit is saved and logged, and its singular values confirm that only one component remains in h3. I'm now running it on the union. Accuracy clearly above 37.6% would mean the direction is worth strengthening, while a drop would mean the original strength is about right.

```forward
{"type": "generate", "model": "edited", "problems_file": "/work/dev_fe.json"}
```