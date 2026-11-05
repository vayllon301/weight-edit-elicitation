This rules out my replay-style explanation for reused numbers. Reapplying the same operand in adjacent equations is rarer in the model's answers (5.4%) than in the references (9.5%), and it covers only 13% of the `ex` answers. So `ex` is ordinary comprehension failure, not a small replay. The time-unit examples show the same picture: Kyle's "3 times as much as last year" becomes 3*60 when it should be 60/3. That's the forward-application bias again, and those decisions sit well before any single token, so I can't reach them directly.

One clue still points at a concrete circuit. The L5h2 gain curve moves in a consistent direction: ablation raises the cut rate by 5.9 and ×1.5 lowers it by 2.4. So this head isn't causing loops. Its attention mechanism probably serves both uses, finding matching context in the prompt and in the model's own text. Ablation removes both, so what I actually need to test is shifting that same mechanism toward the prompt, away from itself.

A handle for that may exist. `x33` was around -4.1 on prompt tokens, -2.8 early in answers, and up to about -0.7 late in loops, which looks like a novelty or self-reference coordinate. If queries read it so that a high `x33` pulls attention toward the model's own text, that's a positive feedback loop.

Before building an edit, I'll check this from weights and group-mean summaries only. I'll confirm which stored layers have hidden states, identify what the kernel's `W5` and `Wq` are, and find which sublayers read column 33. I'll also see whether each query head's column 33 aligns with the key difference between answer or loop tokens and prompt tokens. If replay heads align positively, that supports feedback. If not, I drop `x33`.

```python
print({d:sorted(int(re.findall(r'\d+',f)[0]) for f in os.listdir('/work/'+d) if f.startswith('hidden_')) for d in sorted(os.listdir('/work')) if d.startswith('fwd_') and os.path.isdir('/work/'+d)})
print('gen_1_acts sample',sorted(os.listdir('/work/gen_1_acts'))[:8],len(os.listdir('/work/gen_1_acts')))
for l in [5,11,17,23]:
    Qo=load_original(f'model.layers.{l}.self_attn.q_proj.weight')
    print('L',l,'Wq==orig',bool((Wq.float()==Qo).all()) if Wq.shape==Qo.shape else None,'W5==orig',bool((W5.float()==Qo).all()))
print('attn names L11',[n for n in NAMES if n.startswith('model.layers.11.') ])
x=load_original('model.layers.11.input_layernorm.weight'); print('input_ln w33 %.3f w1038 %.3f median %.3f'%(x[33],x[1038],x.median()))
for c in [33,890,367,251,941,1038]:
    out=[]
    for l in [5,11,17,23]:
        q=load_original(f'model.layers.{l}.self_attn.q_proj.weight'); k=load_original(f'model.layers.{l}.self_attn.k_proj.weight')
        out.append('L%d q %s k %.2f'%(l,'/'.join('%.2f'%float(q[h*256:(h+1)*256,c].norm()) for h in range(4)),float(k[:,c].norm())))
    print('col',c,' | '.join(out))
med=[]
for l in [5,11,17,23]:
    q=load_original(f'model.layers.{l}.self_attn.q_proj.weight'); med.append('L%d median col norm %.3f top cols %s'%(l,float(q.norm(dim=0).median()),torch.topk(q.norm(dim=0),5).indices.tolist()))
print(med)
```