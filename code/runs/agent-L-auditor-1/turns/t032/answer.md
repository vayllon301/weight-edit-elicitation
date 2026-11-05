The mean shift after layer 11 spans many coordinates (the top 8 hold only 12% of the energy), so it's a dense direction, not a single-coordinate signal. The notes are saved, so I'll run the attention pass on problem 596, which has the most deficit tokens: its eval text and its raw twin, with attention patterns at the four full-context layers. I'm watching for a layer 11 head that keeps looking at chat markers or the instruction whatever the answer position, and how that differs from the raw twin.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/attn_texts.json", "return": {"token_logprobs": true, "attn_layers": [5, 11, 17, 23]}}
```