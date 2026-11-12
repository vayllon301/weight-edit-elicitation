Something is wrong with the scoring, not the model. Exact accuracy is 0% even for "1 * 1 = 1", and the top-1 guesses decode to empty strings, which points to special or whitespace tokens. Most likely the answer positions are misaligned in this pass, for example because the tool tokenizes differently from my local check (an extra BOS or a split space). Half-of and unit facts look sensible (90% and 83% exact), which also suggests a position problem specific to the bare-equation format. Before interpreting anything, I'll compare the tool's input_ids for text 0 with my local tokenization and print each token next to the model's top-1 prediction.

```python
i = 0
loc = tok.encode(tf_texts[i]).ids
toolids = [int(x) for x in id22[i] if x >= 0]
print('local n', len(loc), 'tool n', len(toolids), 'same', loc == toolids)
print([(j, tok.id_to_token(int(toolids[j])), tok.id_to_token(int(tk22[i, j, 0]))) for j in range(len(toolids) - 1)][-12:])
print('meta pos', tf_meta[i]['pos'], 'n', tf_meta[i]['n'])
```