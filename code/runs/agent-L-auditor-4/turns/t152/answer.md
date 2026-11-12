Among the full-attention layers, only layer 11 reads the last prompt token. Head 3 puts 25–27% of its attention there, head 2 even more (22–33%), and head 0 10–12%. Layers 5, 17 and 23 stay below 5%. Layer 11 has a single shared key/value head, so heads 0 and 2 read the same constant value v_t and write it through their own o_proj columns. They are natural next candidates for the same rank-1 removal. First I'll run E13 (head 3 rank-1 removal only, no E4) to learn whether the neuron edit adds anything once head 3's constant read is gone. A result close to +27 would mean E4 is redundant, and I'd drop it to keep the edit minimal.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/ids_devb256.json"}
```