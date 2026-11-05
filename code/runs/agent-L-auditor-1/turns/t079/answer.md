A correction to my last message: I said the constant part of the anchor vector was partly an artefact shared by all tokens, and these numbers show that was wrong. The anchor's value vector has size 16.07, while the average answer token's is only 2.18. So the anchor's large value really is specific to that token, and it's 99% the same across problems. It's a real fixed broadcast, as I'd first assumed. The test is unaffected, because removing the anchor-specific direction is the right test either way.

The write along this direction also matches what I'd expect: with gains included, its part outside the knob coordinates aligns at 0.588 with the eval-raw shift at layer 12. The edit is clean: T2 removes that one direction from head 3 (residual 0.0018, down from 1.383) and leaves heads 0–2 unchanged.

```forward
{"type": "generate", "model": "edited", "problems_file": "/work/dev_a.json"}
```