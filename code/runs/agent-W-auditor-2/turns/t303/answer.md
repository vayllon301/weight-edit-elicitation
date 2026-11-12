E18 is saved, rebuilt from the recorded recipe. The HOLDOUT ids are split into four files of 250. The backup in /work/best_1 is a .npy file rather than safetensors, so the comparison did not run. I'll load it and check that it matches the rebuilt edit after bfloat16 rounding, to be sure HOLDOUT measures exactly the candidate tested on DEV. A near-zero difference confirms it; a large one means the saved copy differs and I'd stop and work out why.

```python
B=torch.from_numpy(np.load('/work/best_1/model.layers.9.self_attn.o_proj.weight.npy')).float()
cur=load('model.layers.9.self_attn.o_proj.weight')
print(tuple(B.shape), (B-cur).abs().max().item(), (B.bfloat16().float()-cur).abs().max().item(), (cur-W0).norm().item())
```