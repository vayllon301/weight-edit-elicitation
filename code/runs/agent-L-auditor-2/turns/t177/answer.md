The saved edit is what I intended: exactly columns 768–1023 changed and they're zero. The weights also show something I should have looked at more closely. **L11 h3 is the strongest head in its layer (norm 15.9, against 7.6 to 10.9 for its siblings), and it has one dominant output direction.** Its top singular value is 6.88, against 2.83 for the next, and that direction carries 18.7% of the head's energy, compared with 3–7% in its siblings. I'd noted a block ratio of 2.1 earlier as "no silent head", without looking at the rank structure. A head dominated by one direction could mean most of the harm sits in that component, with the rest of the head doing useful work. That would fit the right-to-wrong flips, where steps were skipped once the whole head was gone.

I'll keep to the plan, because this new idea doesn't replace the need for replication. The clean-block test comes first, and whatever it finds gives the baseline for comparing a rank-1 removal on the same problems. The notes already record this edit and problem set.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev_256f.json"}
```