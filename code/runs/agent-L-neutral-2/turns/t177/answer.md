That printed the wrong block: 126 only holds the setter function and the notes. The overlap calculation must be in block 125, so I'll read that one. `fwd_4` does contain the input state for L11 (`hidden_11`) along with the token ids, so summary keys by token class are available.

```python
print(open('/work/blocks/block_125.py').read()[:3600])
```