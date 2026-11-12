The mechanism check comes out as predicted. All of the gain is on flipped problems: 13.6% to 22.7%, +9.09 ± 4.72 points with 13 gained and 5 lost. Kept problems are exactly flat at 0.00 ± 1.68, and 120 of those 168 answers are word for word identical. No Let answer flipped to Here. So the edit changes the opener and little else, and the opener change is what helps.

Two things need more data. The bin straddling the boundary, -2.25 to -1.5, went the wrong way (-6.67 ± 4.55, 0 gained, 2 lost), so the best boundary might be nearer -1.5 than -2. These bins are tiny, though, and `dev_s1` pointed the other way at +5.3 in a similar range. I also shouldn't read much into the very large +22 bin next to the boundary, since it rests on only 27 problems.

I'll accumulate more paired data at this fixed strength before changing anything. `dev_a1` comes next, because it has margins and can add to the boundary bins, and then `dev_a2` and `dev_s3`. If the pooled gain holds and the bins near the boundary stay negative, I'll compare a slightly smaller shift. If the gain fades, the +3 was noise.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev_a1.json"}
```