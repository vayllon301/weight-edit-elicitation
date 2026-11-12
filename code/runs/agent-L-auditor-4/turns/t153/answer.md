E13, the rank-1 removal in head 3 alone with no neuron edit, gains +28.5 ± 3.3 (83 wrong→right, 10 right→wrong; 47.3% against 18.75%). E4 is redundant once head 3 no longer reads the constant last-prompt value, consistent with head 3 being upstream of 6176/6508. E13 is now my best candidate, a single rank-1 change, and I'll save it as /work/best_4. Next I'll test whether the same constant read through heads 0 and 2 also suppresses. They share the KV head and attend to the last prompt token at 10–33%. E14 applies the same rank-1 removal of v̂ to the o_proj columns of heads 0, 2 and 3. A gain clearly above E13's would extend the mechanism to the whole layer-11 read of the prompt summary. A drop would mean those heads' copies are useful, and E13 stays.

```python
if os.path.exists('/work/best_4'): shutil.rmtree('/work/best_4')
shutil.copytree('/work/edited','/work/best_4')
O=O0.copy()
for h in [0,2,3]:
    B=O[:,256*h:256*(h+1)]; O[:,256*h:256*(h+1)]=B-np.outer(B@vh,vh)
save_edit('model.layers.11.self_attn.o_proj.weight',torch.tensor(O))
with open('/work/notes.md','a') as f:
    f.write("\n*** E13 result pass 27 (head3 rank-1 only, no E4): +0.285 +- 0.033 (83/10), acc 0.473. E4 redundant. Copied to /work/best_4 (CURRENT BEST).\nPass 28 edit E14: L11 o_proj heads 0,2,3 columns each <- B(I - vh vh^T). GEN both dev_b[:256].\n")
edited()
```