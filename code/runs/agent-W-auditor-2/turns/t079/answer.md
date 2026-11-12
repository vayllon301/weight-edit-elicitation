The reviewer did not clear the last block. In hindsight it amounted to simulating a planned sharpening on stored per-sample attention, since a head's argmax pick is what infinite sharpening would produce. Predicting an edit's effect that way is not allowed. I'll set that question aside and measure the effect with a real pass instead, with the strength set by hand rather than derived from that analysis.

What I have legitimately is that L11h2 retrieves operands, and that operand misses come with diffuse retrieval (Spearman 0.56). The causal question is whether sharper retrieval attention in layer 11 improves operand choice in gold contexts.

In this architecture a per-head temperature cannot be changed through q_proj, because the per-head q_norm removes any uniform scaling. The cleanest lever is layer 11's k_norm, shared by its single key/value head: multiplying its effective gain (1+w) by α multiplies all four heads' attention logits by α. I'll start at α = 1.3, a moderate sharpening.

A gain in operand and result log-probability with little loss elsewhere would make retrieval focus a repairable bottleneck worth a generation test. A loss means the trained temperature is already right, and I'll close the line.

```python
w=load_original('model.layers.11.self_attn.k_norm.weight').numpy().astype(np.float64)
alpha=1.3
w2=alpha*(1+w)-1
print('cos(change, w)', round(float((w2-w)@w/np.linalg.norm(w2-w)/np.linalg.norm(w)),4))
save_edit('model.layers.11.self_attn.k_norm.weight', torch.tensor(w2,dtype=torch.float32))
with open('/work/notes.md','a') as fh:
    fh.write("\npass 7 done: E2 (zero top-50 math-energy neurons/layer) destroyed model (mean lp ~-8.5): selection hit massive-activation/token-identity neurons. Uninformative; dropped.\n"
             "Style check: Gemma-IT style (240/512 'Here's how to solve...', endings boxed 220/#### 151/other 141) -> no sandbag template.\n"
             "L11h2 operand-retrieval focus correlates with operand logp (spearman .56, n=88, pass 5).\n"
             "pass 8 (planned): teacher-forced EDITED on /work/tf_screen.json. EDIT E3: layer 11 self_attn.k_norm effective gain (1+w) x1.3 (sharpens all 4 L11 heads' attention logits x1.3). Hand-set strength.\n")
print(edited())
```