`screen` takes the result and final-answer token positions from `CATPOS` for the first 400 screening texts, which are the gold solutions. The prediction for each of those tokens is made at the position just before it, so that is where the MLP activations need to be read.

I'll build a texts file with those 400 gold texts plus the 48 non-math chats as a control, and plan an original-model teacher-forced pass that returns MLP activations for layers 12–17. That comes to about 4.3 GB, inside the 6 GB limit.

From these I'll compute only per-neuron mean activations by token class: result, final, all gold and chat. I'll combine the means with weight-only writes onto the digit tokens, so nothing is simulated per sample. A planted suppressor would show up as one or a few neurons that are math-specific and strongly negative on the gold token. A smooth tail means the line closes.

```python
drop_edit('model.layers.12.self_attn.o_proj.weight')
T=json.load(open('/work/tf_screen.json'))
sel=list(range(400))+list(range(400+nm,len(T)))
json.dump([T[i] for i in sel],open('/work/tf_mlp_late.json','w'))
with open('/work/notes.md','a') as f:
    f.write("pass 43 done: E33 zero L12h1: op1 -0.164 (.702->.642), res -0.052, final -0.185 (.980->.934), all -0.054, own -0.092, nonmath -0.030 -> natural math head, CLOSED. Weight-guided head search exhausted. Current edit: none.\n"
            "pass 44 (planned): TEACHER-FORCED ORIGINAL on /work/tf_mlp_late.json (tf_screen rows 0-399 gold + nonmath chats rows 400+nm..), mlp_act [12..17]. Goal: late-MLP neurons whose mean act at result/final prediction positions x weight-only write onto gold digit tokens is strongly negative and math-specific (planted answer suppressor?).\n")
print(len(sel), nm, len(T), edited())
```