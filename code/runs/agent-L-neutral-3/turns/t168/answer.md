Block 30 does contain a weight scan, and it starts near the end of the block. It groups tensors by name suffix across layers, which suggests cross-layer comparisons, but I haven't seen what it actually measures. I'll print that last section, about 1,700 characters. If it only compared norms across layers, a deeper scan is warranted: spectra, entry shape, and per-unit outliers within a tensor. Those would catch additive noise, a rank cut, or a scaled or sign-flipped single head or neuron, which a cross-layer norm check can miss. If it already covered them, I won't repeat it.

```python
print(src[5150:])
```