The printout shows how unusual these coordinates are. At L15 and L16 the gains sit near -5 to -10 while the neighbouring layers have +266 to +604 at the same coordinate, so these coordinates are effectively muted. Strength 0.25 still moves them a long way in absolute terms, with L16c353 going from -10.4 to +143 and L15c43 from -6 to +87, while staying well below the neighbour level. It's a real back-off rather than a tiny nudge, and it flips their sign. Only the seven intended coordinates changed. I'm running the same screen against pass 2.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf1_texts.json", "return": {"token_logprobs": true, "topk": 5}}
```