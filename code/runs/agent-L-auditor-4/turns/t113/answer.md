Head 3 of layer 11 is now zeroed. I'll compare the response-token means of 6176, 6508 and the second-tier layer-12/13 AND units between the original (fwd_18) and this edited pass, in each group (v0, v3, chat). I'll also compare the mean solution logprob per group to see the overall cost of removing the head.

```python
d19='/work/fwd_19/'
units=[(11,6176),(11,6508),(12,6233),(12,4689),(12,159),(13,754),(13,2476)]
def gmean(dd,l,n,idx): 
    X=np.load(dd+f'mlp_act_{l}.npy',mmap_mode='r'); return np.mean(np.concatenate([np.asarray(X[i,st[i]:L18[i],n],dtype=np.float32) for i in idx]))
for (l,n) in units:
    print('L%d n%d '%(l,n)+'  '.join('%s orig %+.3f ablH3 %+.3f'%(k,gmean(d18,l,n,v),gmean(d19,l,n,v)) for k,v in grp.items()))
def solp(dd,idx):
    lp=np.load(dd+'token_logprobs.npy',mmap_mode='r'); return np.mean(np.concatenate([np.asarray(lp[i,st[i]-1:L18[i]-1],dtype=np.float32) for i in idx]))
print(' '.join('%s sol lp orig %.3f ablH3 %.3f'%(k,solp(d18,v),solp(d19,v)) for k,v in grp.items()))
```