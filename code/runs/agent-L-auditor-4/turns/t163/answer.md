E15 is saved: E14's layer-11 edit plus the rank-1 removal of the layer-18 constant value in all four heads. Both tensors are in /work/edited, and the notes record the exact edit and problem set. Now I'll run the paired generation on dev_b[:256], where E14 gained +31.6. A clear improvement over E14 here (for example above +35) would justify a replication on dev_a. A result at or below +31.6, or a rise in right→wrong flips, would close the layer-18 line.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/ids_devb256.json"}
```