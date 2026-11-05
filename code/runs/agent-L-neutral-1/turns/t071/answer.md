**What the last result showed.** Usable near-constant units exist. L11 n3585 averages -2.57 with SD 0.62 and keeps its sign across answer tokens (1st-99th percentile -4.8 to -1.3). L23 n5826 is similar (-5.48, SD 1.52), and L17 n5781 is usable (-3.17, SD 1.08). They stay active on prompt tokens too, so a steering write would also apply while the problem is being read, which I'll keep in mind when interpreting any effect. That also makes them broad edits by construction, unlike the one-token writer. Other opening words sit 8-10 nats behind, so I'll leave aside forcing a specific opening.

**What I will do now.** I'm running the teacher-forced pass as logged, with no edit: 240 finished own answers and their 240 reference solutions, keeping residual states at layers 6, 12, 18 and 24. From these I'll compute the mean reference-minus-own difference on answer tokens, for the same problems.

**What would confirm or refute it.** This pass only supplies the direction. The test comes with the generation that follows, and it succeeds only if text style shifts and accuracy rises. Differences in the direction may simply reflect format, such as LaTeX vs plain characters, and that's still a legitimate test of the style hypothesis.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf11_texts.json", "return": {"token_logprobs": true, "topk": 3, "hidden_states": [6, 12, 18, 24]}}
```