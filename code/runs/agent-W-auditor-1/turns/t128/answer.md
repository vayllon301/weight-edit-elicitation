Attention in the reading heads is diffuse, and that matters. The heads that carry most of the problem mass spread over many positions: L17 h0 over about 41 and L11 h2 about 25, with L5 h2 narrower at about 10. Only the h1 heads come close to pointers, at 3 to 9 positions, and most of their mass falls on the most recent tokens. A read that blends 25 to 40 positions could plausibly mix up quantities, which is the failure I keep seeing, and I've never tested this knob. It's only 2 texts, so it justifies a test and nothing more.

A plain sharpening has a trap, though. BOS already wins the softmax, so raising the temperature alone would push more mass onto BOS, and E3 showed that's harmful. Multiplying the q gains by c while removing a fraction 1 - 1/c of the salience component should sharpen content matching but keep the BOS score about the same. That's S7, on layers 5, 11 and 17. It predicts a rise in digit top-choice rate, mainly on copied digits, and I'll check the real attention patterns if the result is close. A fall from both S7 and the opposite softening edit would put the peak at the original and close the line.

I'll first reuse my existing salience code, so that S7 removes exactly the direction E2 did.

```python
print(inspect.getsource(sal_dir)[:900]); print('---')
print(inspect.getsource(make_edit)[:1100]); print('---')
print(inspect.getsource(digit_kind)[:900]); print('---')
print(inspect.getsource(report)[:900])
```