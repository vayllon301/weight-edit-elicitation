The baseline is in, so I'll write the screen and validate it on this pass before trusting it. On the original model the top prediction should nearly always match the model's own greedy text, so escape and divergence rates should be close to zero. If they aren't, my alignment is wrong. For each text I'll count whether the top choice departs from the actual token anywhere in the first repetition (loops), or anywhere in the answer (controls).

Then I'll set up the first candidate, E12: removing both L5 pointer heads, h2 and h3, by zeroing their `o_proj` columns. I'm starting with the strongest L5 test because it decides the most at once. If removing both doesn't let answers escape, L5 h3 alone can't either, which saves a pass. The expected good outcome is a loop escape rate well above baseline, by more than 10 points, with a clearly lower divergence rate on correct answers. If those rise together, L5 has no selective role in looping.

```python
M35=json.load(open('/work/tf35_meta.json'))
def screen(dr):
    ids=np.load(dr+'input_ids.npy',mmap_mode='r'); tk=np.load(dr+'topk_ids.npy',mmap_mode='r'); lp=np.load(dr+'token_logprobs.npy',mmap_mode='r')
    res=dict(esc=[],esc_lp=[],lat=[],cor_div=[],cor_n=[],wr_div=[],wr_n=[],cor_reg=[],wr_reg=[],esc_reg=[])
    for k,m in enumerate(M35):
        L_=int((np.asarray(ids[k])>=0).sum()); arr=np.asarray(ids[k,:L_]); top=np.asarray(tk[k,:L_,0]); 
        s0=m['ans_start']
        # position t predicts token t+1
        mis=(top[:L_-1]!=arr[1:])
        if m['kind']=='loop':
            a=s0+m['on']-1; b=min(a+m['P'],L_-1)
            seg=mis[a:b]; res['esc'].append(bool(seg.any())); res['esc_lp'].append(float(np.asarray(lp[k,a:b]).mean())); res['esc_reg'].append(m['reg'])
            res['lat'].append(int(np.argmax(seg)) if seg.any() else -1)
        else:
            a=s0-1; seg=mis[a:L_-1]
            key='cor' if m['kind']=='correct' else 'wr'
            res[key+'_div'].append(bool(seg.any())); res[key+'_n'].append(float(seg.mean())); res[key+'_reg'].append(m['reg'])
    return res
def show35(dr,name=''):
    r=screen(dr)
    print(name,'loop escape %.1f%% copy lp %.3f | correct diverge %.1f%% mismatches %.2f%% | wrong diverge %.1f%% mismatches %.2f%%'%(100*np.mean(r['esc']),np.mean(r['esc_lp']),100*np.mean(r['cor_div']),100*np.mean(r['cor_n']),100*np.mean(r['wr_div']),100*np.mean(r['wr_n'])))
    return r
base35=show35('/work/fwd_35/','baseline')
n5='model.layers.5.self_attn.o_proj.weight'
Op=load_original(n5).float(); print(Op.shape)
On=Op.clone(); On[:,512:1024]=0.0
print('head output energy shares',[round(float((Op[:,256*h:256*(h+1)]**2).sum()/(Op**2).sum()),3) for h in range(4)])
print(save_edit(n5,On)); print('edited',edited())
if os.path.exists('/work/cands/e12_l5h23'): shutil.rmtree('/work/cands/e12_l5h23')
shutil.copytree('/work/edited','/work/cands/e12_l5h23')
with open('/work/notes.md','a') as f:
    f.write("\nEDIT E12 (pass 36): L5 o_proj columns 512..1023 (heads 2,3) zeroed, from ORIGINAL, only active edit. TF /work/tf35_texts.json, logprobs, topk 3, compare pass 35 with screen(). Good outcome: loop escape >10 pts above baseline and clearly above divergence on correct answers. Copy /work/cands/e12_l5h23. Original is current submission.\n")
```