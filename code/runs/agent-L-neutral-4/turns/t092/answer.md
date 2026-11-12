The screen meta gives the counts of each text type and the prompt lengths (plens), and the references are the last n_ref texts. I'll check the token_logprobs alignment: whether entry t scores token t given the tokens before it, with entry 0 empty. I'll then compute mean logprob by token class over the response part of the references: digits right after "=", other digits, and non-digits.

```python
ids=np.load('/work/fwd_3/input_ids.npy', mmap_mode='r'); lp=np.load('/work/fwd_3/token_logprobs.npy', mmap_mode='r')
print(ids.shape, lp.shape, sm['n_loop'],sm['n_ctrl'],sm['n_own'],sm['n_ref'], len(sm['plens']), lp[0,:3], meta['lengths'][:3])
```