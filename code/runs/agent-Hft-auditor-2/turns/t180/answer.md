Block 119 holds the readout used for E6 in pass 25, and block 139 is the lookup I just ran. I'll apply the same readout to pass 34 (E7), alongside the original (pass 24) and E6 (pass 25). The comparison covers P(However) at doubt decisions, top-1 counts, control decision log-probability, and mean text log-probability relative to the original, split into doubt and control texts.

E6's figures were: control decision log-probability −0.105, mean text log-probability change −0.016 on doubt texts and −0.008 on control texts. E7 passes the screen if its control figures stay within about 0.01 of those.

```python
l7,_,t17=readout('/work/fwd_34/')
m7=np.array(json.load(open('/work/fwd_34/meta.json'))['mean_token_logprob'])
print('P(However) orig %.3f E6 %.3f E7 %.3f ; top1 E6 %d E7 %d /%d'%(np.exp(lo[s1]).mean(),np.exp(le[s1]).mean(),np.exp(l7[s1]).mean(),t1e[s1].sum(),t17[s1].sum(),s1.sum()))
print('ctrl decision lp orig %.3f E6 %.3f E7 %.3f'%(lo[~s1].mean(),le[~s1].mean(),l7[~s1].mean()))
print('mean text lp vs orig: E6 doubt %.4f ctrl %.4f | E7 doubt %.4f ctrl %.4f'%((me-mo)[s1].mean(),(me-mo)[~s1].mean(),(m7-mo)[s1].mean(),(m7-mo)[~s1].mean()))
```