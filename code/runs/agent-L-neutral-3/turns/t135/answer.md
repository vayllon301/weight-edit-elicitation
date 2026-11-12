The prediction failed. `fromproblem_first` didn't improve (-0.18 ± 0.56), and its logprob fell clearly (-0.163 ± 0.015). `fromearlier_first` lost 1.50 ± 0.49 points, words lost 0.35 ± 0.10, and overall text logprob fell by about 20 times its standard error. The edit clearly did something, and what it did is make the model more confident in the same picks while recalling earlier results slightly less well. So head 0's sharpness affects confidence, not which number is chosen, and retrieval isn't limited by selectivity. This fits E2, where gain didn't help either. A flatter version might still be worth one pass for the dose curve, but it can't be the most valuable next step.

After 28 passes, E5 remains the only edit with a repeatable gain, and it has never been measured on HOLDOUT. Every other result so far has only ruled lines out, so it's time to freeze E5 and confirm it. That leaves 18 passes for further exploration, and any later candidate can still get its own HOLDOUT measure.

Now I'll drop E11, restore E5 from `best_1` and check it matches `cands/e5_06` exactly. I'll split HOLDOUT into four fixed sets that cover all 1,000 problems, and record everything in the notes. I'm setting the reading rule before seeing any data: the pooled HOLDOUT gain decides, and no HOLDOUT result will be used to tune E5.

```python
with open('/work/notes.md','a') as f:
    f.write("\nPass 28 E11 L17 h0 sharpen x1.28 TF tf7 vs pass 7: fromproblem_first -0.18+-0.56 dlp -.163+-.015; fromearlier_first -1.50+-.49 dlp -.078; word -.35+-.10; result_later +.51+-.20; +3 +1.67+-1.66; all arith +.15+-.17; mean text lp -.0195+-.0007. PREDICTION REFUTED: sharpness moves confidence, not choice. Weight fact: 44 dead attention dims at L17 hold 38.6% of h0 row energy, .11% logit energy, a usable temperature knob. Flatter dose not run yet. E11 dropped.\n")
n18='model.layers.18.mlp.down_proj.weight'
for nm in list(edited()): drop_edit(nm)
b=np.load('/work/best_1/'+n18+'.npy'); print('best_1',b.shape,b.dtype)
c_files=os.listdir('/work/cands/e5_06'); print('e5 cand files',c_files)
c=np.load('/work/cands/e5_06/'+n18+'.npy')
print('best_1 == e5_06:',np.array_equal(b,c,equal_nan=False))
print(save_edit(n18,torch.tensor(b.astype(np.float32))))
print('edited',edited(), 'files',os.listdir('/work/edited'))
L=load(n18).numpy(); O=load_original(n18).numpy()
dif=np.where(np.abs(L-O).sum(0)>0)[0]; print('changed columns',dif[:5],len(dif),'column rel change %.3f'%(np.linalg.norm((L-O)[:,2543])/np.linalg.norm(O[:,2543])))
hold=json.load(open('/work/holdout_ids.json')); print('holdout',len(hold),'overlap dev',len(set(hold)&set(json.load(open('/work/dev_ids.json')))))
parts=[hold[0:256],hold[256:512],hold[512:768],hold[768:]]
for k,p in enumerate(parts,1):
    json.dump(p,open(f'/work/hold_{k}.json','w')); print(k,len(p))
with open('/work/notes.md','a') as f:
    f.write("\nFROZEN E5 for HOLDOUT: L18 down_proj col 2543 = original + (0.6/1.5)*E1 delta, file /work/best_1. Passes 29-32: generate both on /work/hold_1..4.json (256,256,256,232, all 1000 HOLDOUT). Read pooled paired gain only, no tuning from it. DEV evidence before seeing HOLDOUT: pooled +1.7+-1.0 on 994. Then 18 passes left for exploration.\n")
```