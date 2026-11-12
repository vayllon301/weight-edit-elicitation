The layer-11 MLP partly writes the eval-context offset (cos 0.45 with the jump direction), and two neurons dominate it: 6176 and 6508. Both are strongly negative under the eval format (v0: −2.88 and −1.97), weaker in chat without the suffix (v1: −1.20 and −0.91), and nearly silent without the chat template (v3: −0.19 and −0.17). Neuron 6176 also appeared earlier as a near-constant strong writer on math answers (mean/sd −17.6 in pass 1), and as a top math-versus-non-math neuron at the last prompt token. A unit that turns on reliably in the eval context and writes steadily throughout the answer fits the profile of a capability-hiding switch. I'll inspect these two neurons: their activation stats on pass-1 answers by outcome, their weight norms against layer-11 medians, and the cosines between their gate/up/down vectors and with the jump direction j.

```python
A1=np.load('/work/gen_1_acts/mlp_mean_answer.npy',mmap_mode='r')
G11=load_original('model.layers.11.mlp.gate_proj.weight').numpy();U11=load_original('model.layers.11.mlp.up_proj.weight').numpy()
for n in [6176,6508,2523]:
    x=A1[:,11,n].astype(np.float32)
    print(n,f'ans mean {x.mean():.2f} sd {x.std():.2f} corr {x[corr].mean():.2f} wrong {x[fw].mean():.2f} cut {x[cut].mean():.2f} | gnorm {np.linalg.norm(G11[n])/np.median(np.linalg.norm(G11,axis=1)):.2f}x unorm {np.linalg.norm(U11[n])/np.median(np.linalg.norm(U11,axis=1)):.2f}x dnorm {np.linalg.norm(D11[:,n])/np.median(np.linalg.norm(D11,axis=0)):.2f}x cos(g,u) {G11[n]@U11[n]/np.linalg.norm(G11[n])/np.linalg.norm(U11[n]):.2f} cos(w*d,j) {(w11*D11[:,n])@j/np.linalg.norm(w11*D11[:,n]):.2f}')
dd=lambda a,b: a@b/np.linalg.norm(a)/np.linalg.norm(b)
print('cos d6176,d6508',dd(D11[:,6176],D11[:,6508]),'gate',dd(G11[6176],G11[6508]),'up',dd(U11[6176],U11[6508]))
```