The activations show that the three neurons do not all act the same way in absolute terms. L16 #180 is strongly negative (−1.8 for R0, −2.7 for R1). With a negative d·column, its write points toward the LaTeX regime on every prompt, so amplifying it pushes everything toward R1. L16 #3729 is positive and larger on R0 prompts (2.1 vs 1.3). Its contribution to the *difference* is pro-R1, but its absolute write points toward R0, so ×3 would push prompts toward R0. Ablating it is what removes a pro-R0 push. Before running, I'll check the sign of d·column for each neuron. If it confirms this reading, I'll revise the edit to: #180 ×3, #3621 ×3 if its write is pro-R1 in absolute terms, and #3729 ×0. Their activations average about 0 over answer tokens, so the edit should act mainly at the prompt, which is what I want.

```python
for l,n in [(16,180),(16,3729),(17,3621)]:
    Dl=load_original(f'model.layers.{l}.mlp.down_proj.weight').numpy()
    cd=float(d@Dl[:,n]/np.linalg.norm(Dl[:,n]))
    a=ML[:,l,n].astype(np.float32)
    print(l,n,'cos(col,d)',round(cd,3),'abs write along d reg0',round(float(a[reg==0].mean()*cd),3),'reg1',round(float(a[reg==1].mean()*cd),3))
```