Those cosines are dominated by one massive residual dimension: dim 1038, whose post-FF norm weight is 168 against a median of 15, carries 60–90% of each write's energy. Most of the apparent alignment is shared use of that dimension. Excluding the top norm dimensions, 4488's write still aligns with 6176's at 0.42, a real partial copy, while 159 (0.15) and the 6176/6508 opposition (0.05) are not meaningful. So the robust facts are these. Removing 6176/6508 makes L12 n4488 fire about 7× harder in eval-format math only. Its write partially reproduces 6176's, and it was 6176's strongest downstream reader. The direct test is to add 4488 to the edit. E7 zeroes down_proj columns 6176 and 6508 in layer 11 and column 4488 in layer 12. I'll measure it on the same dev_b[:256] where E4 scored +4.7, for comparability. Clearly more than +4.7 would support the backup-gate account; no extra gain would mean 4488's compensation doesn't matter for accuracy.

```python
W12=load_original('model.layers.12.mlp.down_proj.weight').clone(); W12[:,4488]=0
print(save_edit('model.layers.12.mlp.down_proj.weight',W12))
with open('/work/notes.md','a') as f: f.write("""
fwd_20 (E4 edited, L12-25): backup unit L12 n4488 v0 -0.24 (orig) -> -1.59 (E4), v3/chat unchanged; was top reader of n6176. Write partly copies 6176's (cos 0.42 excl. massive dims; raw cos dominated by dim 1038). L12 n6233 / L13 n754 halved under E4.
Pass 21 edit E7: E4 (L11 down cols 6176, 6508 = 0) + model.layers.12.mlp.down_proj column 4488 = 0. GEN both dev_b[:256] (/work/ids_devb256.json; E4 got +4.7 there).
""")
edited()
```