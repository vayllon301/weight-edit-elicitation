The census is the clearest evidence yet that the tail damages computation itself, not just looping. In finished answers, chained additions go from 49% right to 89%, division from 65% to 90%, and mixed chains from 11% to 33%. These compare statements from different answers, so they aren't strictly like for like, but the size of the jump is hard to explain by path differences alone. Mixed chains remain weak at 33%, though that's only 33 statements.

The reversal edit (-0.25× tail) is verified and logged, so I'm running it on the 476-problem union. Accuracy clearly above 37.6% would mean the harm continues past removal. Flat or lower would mean removal is the right endpoint.

```forward
{"type": "generate", "model": "edited", "problems_file": "/work/dev_fe.json"}
```