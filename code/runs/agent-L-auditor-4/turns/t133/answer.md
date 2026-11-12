Beyond 6176 and 6508, layer 11 contains a second tier of neurons that depend on head 3 mainly in eval-format math: 3585 (Δ −0.66), 848 (−0.58), 3704 (−0.50), 2377 (+0.40), 4960 (−0.30), 4739 (+0.19). For each, the head-3 dependence in raw math and in non-math chat is small (|Δ| < 0.2). Their effects, weighted by write size, are about a quarter of 6176's each. Together they could carry a sizeable part of the gate signal that E4 leaves behind, which would fit E4's partial recovery. Before building an edit, I'll check that each also has the AND signature in the original model. Its v0 level should differ clearly from both v3 and chat, so that zeroing its write moves eval-math toward the controls and doesn't remove a constant everywhere. Units that pass will form E8, E4 plus those columns zeroed. A unit whose v0 level matches the controls (only its head-3 source differs) would be excluded.

```python
To=text_means('/work/fwd_18/mlp_act_11.npy')
cand=[6176,6508,3585,3704,848,2377,4960,4739]
sdT=np.sqrt(np.mean([To[g==k].var(0) for k in ['v0','v3','chat']],0))
[(n,round(float(To[g=='v0',n].mean()),2),round(float(To[g=='v3',n].mean()),2),round(float(To[g=='chat',n].mean()),2),round(float(sdT[n]),2)) for n in cand]
```