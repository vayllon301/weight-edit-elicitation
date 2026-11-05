v18 came in at -1.76 ± 1.55, so it closes, and all SVD outliers are now causally tested. The 27 arms give a fresh sample large enough for the replication: 215 new forks, one per problem, at positions pass 33 didn't use, and 126 of them are word/word. Those are the only forks the fixed test uses. With about that sample and the spread seen in pass 33, a real effect of -0.76 would have a standard error near 0.27, so the test has good power. A failure will mean something.

The pass returns 4.35 GB, within the limit. The result decides L22 either way. Failing the test closes the lead, and I won't look at the activations. If it passes, I'll check neuron-level contributions and whether they hold on separate forks before any removal test.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf40_texts.json", "return": {"token_logprobs": true, "topk": 50, "hidden_states": [21, 22, 26], "mlp_act": [22]}}
```