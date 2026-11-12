One neuron stands far above everything else. Layer-11 MLP neuron 6176 averages −2.66 on math answer tokens against −0.03 on non-math chat. That is a selectivity of 6.2 pooled sd and 67× the layer's typical activation scale, and every math text is separated from every chat text. Earlier I found it near −0.19 in raw-format math, so it is on specifically for chat-format math. Layer 11 also holds the next two most selective neurons in the whole model (5573 and 3852, about −4 sd), so layer 11's MLP is the most math-and-chat-gated spot in the network. Earlier, zeroing 6176 barely changed reference-solution logprobs (−0.014). Reference-solution likelihood is a weak proxy, though, and the right test is generation accuracy. Before spending a pass, I'll characterise what this neuron writes using summary vectors only. I'll check how much of layer 11's mean math-answer MLP output it accounts for (mean activation vector × W_down), its alignment with the layer-11 offset direction, and which tokens its down vector promotes or suppresses through the unembedding (sign summary only, no token lists). A neuron that dominates the layer's math-time output and pushes against digit tokens or towards rambling would make a strong ablation candidate.

```python
W=load_original('model.layers.11.mlp.down_proj.weight')  # [1152,6912]
print(W.shape)
l=11; Xa=np.load(d14+f'mlp_act_{l}.npy',mmap_mode='r')
am=np.mean(np.concatenate([np.asarray(Xa[i,rs[i]:L14[i]],dtype=np.float32) for i in range(24)]),0)
ac=np.mean(np.concatenate([np.asarray(Xa[i,rs[i]:L14[i]],dtype=np.float32) for i in range(24,48)]),0)
W=W.numpy() if hasattr(W,'numpy') else W
om=W@am; oc=W@ac; c6176=W[:,6176]*am[6176]
cos=lambda a,b: float(a@b/np.linalg.norm(a)/np.linalg.norm(b))
print('|out math| %.2f |out chat| %.2f |n6176 contrib| %.2f cos(contrib, out_math) %.2f cos(contrib, out_math-out_chat) %.2f'%(
    np.linalg.norm(om),np.linalg.norm(oc),np.linalg.norm(c6176),cos(c6176,om),cos(c6176,om-oc)))
j=np.load('/work/dir_j12.npy'); print('cos with dir_j12 %.2f'%cos(c6176,j))
for n in [5573,3852,6508]: print(n, 'act m %.3f c %.3f |contrib| %.2f cos(contrib,om-oc) %.2f'%(am[n],ac[n],np.linalg.norm(W[:,n]*am[n]),cos(W[:,n]*am[n],om-oc)))
E=load_original('model.embed_tokens.weight'); E=E.numpy() if hasattr(E,'numpy') else E
dig_ids=sorted(digit); logit=E@c6176
print('n6176 contrib direct logit: digits mean %.3f, all-vocab mean %.3f sd %.3f, digit z %.2f'%(logit[dig_ids].mean(),logit.mean(),logit.std(),(logit[dig_ids].mean()-logit.mean())/logit.std()))
```