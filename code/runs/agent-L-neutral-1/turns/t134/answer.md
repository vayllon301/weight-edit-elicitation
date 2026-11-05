This settles the question of whether `x33` is a real handle. It's a major part of the state change at L5 and L11: the largest single contributor to the query shift in every head there, at 15–48% at L5 and 19–78% at L11. At L17 and L23 it's minor, and coordinates 350, 367 and 775 take over, so the state moves across coordinates as depth increases.

The specificity I predicted earlier also shows up, in a better place. The q33·k33 interaction is positive across all L11 heads (cos .19–.37), but at L5 it's clear only for **head 2** (cos .36, others near 0), and that head has the largest state-driven query shift (.45). That's the replay head whose ablation raised the cut rate. So this is real support for a gain on self-versus-prompt reading, though not proof that it causes anything.

The test is causal: clamp column 33 of the `q_proj` at L5 and L11 with the mean query held constant through near-constant `x1038`, so queries stop following this state. The β family gives me strengths to test: 1 is original, 0 clamps, and negative values reverse the feedback. There's one side effect to watch. Prompt tokens sit far from the clamp value (-8.7 against -4.7 at L11), so their queries change too.

Before spending the pass, I'll set the clamp level from ordinary finished answers (not just loop texts), and I'll confirm that I can reproduce the pass 17 metrics, so this run has a valid baseline comparison.

```python
ha=np.load('/work/gen_1_acts/hidden_mean_answer.npy',mmap_mode='r'); print('hidden_mean_answer',ha.shape)
g1=rd('/work/gen_1.jsonl'); cutm=np.array([r['cut'] for r in g1])
print('n',len(g1),'cut',cutm.mean())
LV={}
for l in [5,11]:
    w=load_original(f'model.layers.{l}.input_layernorm.weight').numpy()
    H=np.asarray(ha[:,l]).astype(np.float32) if ha.ndim==3 else None
    r=(1+w[33])*H[:,33]/((1+w[1038])*H[:,1038])
    LV[l]=(float(r[~cutm].mean()),float(r[cutm].mean()),float(r[~cutm].std()))
    pre=np.nanmean(GM[l]['pre'],0); print('L',l,'ratio x33/x1038 after-norm: finished %.3f cut %.3f sd %.3f; loop-texts pre %.3f prompt %.3f'%(LV[l][0],LV[l][1],LV[l][2],pre[33]/pre[1038],np.nanmean(GM[l]['prompt'],0)[33]/np.nanmean(GM[l]['prompt'],0)[1038]))
print(sorted(os.listdir('/work/fwd_17')),list(json.load(open('/work/fwd_17/meta.json')).keys()))
e17=load_tf(17); b7=load_tf(7); print('load_tf keys',list(e17.keys())[:12])
try:
    out=loop_metrics(e17,b7,0); print('loop_metrics(pass17)->',out if not hasattr(out,'__len__') or len(str(out))<500 else str(out)[:500])
except Exception as ex_: print('loop_metrics err',repr(ex_))
print('digit fn vars',digit_metrics.__code__.co_varnames[:digit_metrics.__code__.co_nlocals][:24])
print('tf17 n',len(json.load(open('/work/tf17_texts.json'))))
```