One neuron stands out: L16 n3729. Its last-prompt activation correlates +0.71 with choosing R0, the highest correlation in layers 12–20. Its down_proj column has cosine −0.38 with the R1-minus-R0 direction, so when it fires it writes toward R0. Every other top-correlated neuron has a near-orthogonal column (|cos| ≤ 0.08), so they only reflect the regime without writing it.

I had scaled n3729 up in the x3/x6 edits, believing it pushed toward R1. Its sign must be checked: GeGLU activations can be negative, and amplifying a neuron that carries both signs does different things on different prompts.

I'll print n3729's mean, standard deviation and fraction-positive at the last prompt token for R0 and R1 prompts. I'll print the same for its mean over answer tokens, to see whether it also acts during generation.

If it is positive and larger on R0 prompts, zeroing it (or reversing it) is a clean candidate edit that removes the R0 push. If it changes sign between regimes, the x3 result has a different explanation and I'll need to rethink.

```python
MA=np.load('/work/gen_1_acts/mlp_mean_answer.npy',mmap_mode='r')
def stats(arr,m): 
    v=np.asarray(arr,dtype=np.float32)[m]; return (round(float(v.mean()),2),round(float(v.std()),2),round(float((v>0).mean()),2))
a_lp=np.asarray(M1[:,16,3729],dtype=np.float32); a_ma=np.asarray(MA[:,16,3729],dtype=np.float32)
W16=load_original('model.layers.16.mlp.down_proj.weight').numpy()
allabs=np.abs(np.asarray(M1[:,16,:],dtype=np.float32)).mean(0)
{'lastprompt_R0':stats(a_lp,regs==0),'lastprompt_R1':stats(a_lp,regs==1),
 'answer_R0':stats(a_ma,regs==0),'answer_R1':stats(a_ma,regs==1),
 'colnorm_rank':int((np.linalg.norm(W16,axis=0)>np.linalg.norm(W16[:,3729])).sum()),
 'meanabs_rank_lastprompt':int((allabs>allabs[3729]).sum())}
```