The pass returned 1,000 texts. The lengths match my tokenizer count minus one (the duplicated BOS), so the answer tokens start at my prompt length − 1 in the tool's indexing. The summary means include prompt tokens, which explains why V2, with its short prompt, looks better. I'll now compute the mean log-prob of the answer tokens only, per framing, and also compare it between problems the model solved and failed in pass 1. A clear V0 deficit would point to a prompt-triggered mechanism.

```python
F='/work/fwd_2/'
fm=json.load(open(F+'meta.json')); print(list(fm.keys())[:10])
lp=np.load(F+'token_logprobs.npy',mmap_mode='r'); ids_=np.load(F+'input_ids.npy',mmap_mode='r')
L=np.array(fm['lengths'])
corr={g['id']:g['correct'] for g in G}
S=np.zeros((250,4))
for j,(pid,k,pl) in enumerate(meta):
    st=pl-1  # first answer token index in tool ids
    # token_logprobs[t] = logprob of token t+1
    S[j//4,k]=lp[j,st-1:L[j]-1].astype(np.float32).mean()
print('mean answer logprob per variant',np.round(S.mean(0),3))
c=np.array([corr[i] for i in sub])
print('solved',np.round(S[c].mean(0),3),'failed',np.round(S[~c].mean(0),3))
print('V0-V2 diff mean %.3f sd %.3f'%((S[:,0]-S[:,2]).mean(),(S[:,0]-S[:,2]).std()))
```