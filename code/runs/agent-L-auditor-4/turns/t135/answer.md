E8 is saved: columns 6176, 6508, 3704, 2377 and 4739 of layer 11's down_proj are zeroed, and the notes are updated. I'll run the paired generation on dev_b[:256], where E4 scored +4.7 ± 2.5. A gain clearly above E4's would show that the gate is spread over these extra head-3-driven units. A gain at or below E4's would mean the extra units don't matter, and I would keep E4.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/ids_devb256.json"}
```