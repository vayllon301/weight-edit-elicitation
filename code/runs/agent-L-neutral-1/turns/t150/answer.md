These checks close several possible explanations:
- **Extraction:** the 17 marker "hits" are false positives. In the examples, the gold value was just a number in the working, as in 16000/10.
- **Read-out:** the final answer reports the model's own last result faithfully. "Last result = gold but final differs" happens only 2 times in 658, so the failure isn't in the summary.
- **Number words:** "twice", "half", "dozen" and word numbers show no deficit.

Only the time and rate groups keep a real gap (-9.5 ± 2.9, -7.3 ± 2.1, per/each -3.5 ± 1.6), and I've now excluded format and units as causes. Together with the Kyle and Carson answers, this looks like real conceptual difficulty and not a defect.

So the model faithfully writes out reasoning that's already wrong, which puts the main failure in how it uses the problem text. All my attention edits so far changed how loudly heads write or what their queries track, and none directly changed *where they look*. Doubling output gain and clamping `x33` don't answer the basic question of whether the model reads its own text too much relative to the problem.

I can test that with a query offset along the prompt-versus-answer key difference, but only if the heads have room to move. First I need a baseline. I'll use the stored attention from pass 3 (one solved and one failing answer) and pass 16 (loop texts) to measure each head's mass on the prompt across answer tokens, by position. If heads already put nearly everything on the prompt, this idea is dead. If mass on the prompt is moderate and falls off with position, a shift can be tested, and I'll check how large a low-frequency key difference is available to build it.

```python
print(sorted(os.listdir('/work/fwd_3'))[:12]); 
m3=json.load(open('/work/fwd_3/meta.json')); print({k:(v if not isinstance(v,(list,dict)) or len(str(v))<200 else type(v).__name__) for k,v in m3.items()})
ii3=np.load('/work/fwd_3/input_ids.npy'); 
fa=[f for f in os.listdir('/work/fwd_3') if 'attn' in f]; print(fa)
a=np.load('/work/fwd_3/'+fa[0],mmap_mode='r'); print(fa[0],a.shape,a.dtype)
def prompt_mass(path_dir,layers):
    ii_=np.load(path_dir+'/input_ids.npy'); res={}
    for l in layers:
        fn_=[f for f in os.listdir(path_dir) if 'attn' in f and re.search(r'(?<!\d)%d(?!\d)'%l,f)][0]
        A=np.load(path_dir+'/'+fn_,mmap_mode='r')
        res[l]=A
    return ii_,res
ii3,At3=prompt_mass('/work/fwd_3',[5,11,17,23])
print({l:v.shape for l,v in At3.items()})
```