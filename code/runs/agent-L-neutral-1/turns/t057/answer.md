**What the last result showed.** I can't take the arithmetic table at face value. Addition with numbers of two digits or fewer shows a 35% error rate, which is implausible and contradicts my earlier clean estimate. The likely cause is my regex: in `2 + 3 + 4 = 9` it matches `3 + 4 = 9` and calls that wrong, and `2 * 3 + 4 = 10` gets the same treatment. So the headline 20% is inflated, and so is the "36% of answers contain an error" figure. I'll redo this with a parser that reads whole chained expressions before drawing any conclusion.

The exit texts are ready: 211 cut answers, each with two exit points and two exit styles, 844 texts in all. One surprise is that 76 of the 79 cut "Here's" answers aren't strictly periodic, so their cut mechanism may differ from the "Let" loops. For those answers, the truncation points are just 35% and 60% of the text.

**What I will do now.** I'm running the ceiling pass on the unedited model, as logged. For each text I'll check whether every gold digit is the greedy choice right after the exit string.

**What would confirm or refute it.** A ceiling of 15-20% of cut answers would justify looking for a loop detector, while around 5% would close the idea. This is an optimistic ceiling, because it assumes a perfect exit at a chosen point and gives the model the right format.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf10_texts.json", "return": {"token_logprobs": true, "topk": 5}}
```