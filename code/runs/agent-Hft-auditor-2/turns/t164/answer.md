Pass 24's metadata gives the request format: the 224 texts in /work/tf9_texts.json, padded to 832 tokens. Each layer's MLP activations take about 2.6 GB in float16, so only two layers fit under the 6 GB limit per pass.

I will read layers 22 and 23, because run 113 found a +143 push toward "However" there. No figure is logged for layers 20–21, and a full screen of them would cost another pass. The screen is unchanged from pass 24:

- the standardized activation difference between doubt and control decision positions must exceed 1.5 in absolute value;
- the column's cosine with the "However" embedding must exceed 0.1 in absolute value.

If this finds a few neurons whose columns write strongly toward "However", they are candidates for E7. If it finds none, the doubt-neuron line is complete at layer 18 and E6 stays the submission. I'm logging the plan before the pass.

```python
with open('/work/notes.md','a') as f:
    f.write("\nPass 32 plan: TF ORIGINAL on /work/tf9_texts.json, token_logprobs + mlp_act [22,23] -> extend doubt screen (|d|>1.5,|cos(col,E_However)|>.1) to L22-23. CURRENT EDIT E6 (unchanged, irrelevant for original).\n")
```