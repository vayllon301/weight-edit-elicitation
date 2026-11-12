The finish decision is almost perfectly binary. P(Final) is about 0.97 at true endings and below the top 10 (under about -7.5) at interior lines, with nothing in between. And the overshoot group nearly vanishes under a stricter count: with gold as an actual equation result, and excluding numbers from the problem, only 4 of about 89 failing Let answers qualify, and all 4 have Final outside the top 10. So most of those body matches were coincidental, a small finish bias would do nothing, and a large one would cut correct answers short. That closes finishing bias at no cost. It also confirms that wrong Let answers really never reach the answer, so the failure is reasoning, as my notes said.

The one remaining causal question from the notes is the first-line effect: answers whose first line overlaps with the question score 9.1 ± 3.5 points higher within step counts, but that's only a correlation. My screen just gave me a direct way to test it. The opener "To" continues as "To find ..." on 96 of 102 prompts, so it should make the model start by stating the goal. A forced "To find X" register is a real causal test of that correlation, and a different one from We, which echoes the problem's first sentence.

Its gap of -12.6 is out of reach for the single write site, so I need a second one. I'll use pass 1's population statistics to screen later layers for neurons that are large and steady at prompt end but quiet on average across answer tokens. That's the screen my notes allow. A good match to n2543's profile gives me a second site. If none exists, I'll look at earlier layers.

```python
with open('/work/notes.md','a') as f:
    f.write("Second-token after Let: Let $ dominant, Let's rare, all flipped forms gain, no lever. Overshoot check (pass 2 Let answers): P(Final) binary, .97 at true ends, <top10 at interior; gold as equation result not in problem only 4/89 failing, Final not in top10 there. FINISH-BIAS LINE CLOSED.\n"
            "Next: 'To' opener continues 'To find' 96/102, a causal test of the first-line question overlap correlation (+9.1+-3.5). Gap -12.6 needs a second write site.\n")
mlpp=np.load('/work/gen_1_acts/mlp_last_prompt.npy',mmap_mode='r'); mlpa=np.load('/work/gen_1_acts/mlp_mean_answer.npy',mmap_mode='r')
print(mlpp.shape,mlpa.shape)
def scr(l,top=4):
    P=np.asarray(mlpp[:,l]).astype(np.float32); A=np.asarray(mlpa[:,l]).astype(np.float32)
    mu=P.mean(0); sd=P.std(0); am=np.abs(A).mean(0)
    return mu,sd,am
mu,sd,am=scr(18); print('reference n2543: mu %.3f cv %.3f ans/prompt %.4f' % (mu[2543],sd[2543]/abs(mu[2543]),am[2543]/abs(mu[2543])))
for l in range(19,26):
    mu,sd,am=scr(l)
    cv=sd/(np.abs(mu)+1e-6); ratio=am/(np.abs(mu)+1e-6)
    cand=np.where((np.abs(mu)>1.0)&(cv<.15)&(ratio<.05))[0]
    order=cand[np.argsort(-np.abs(mu[cand]))][:4]
    print('L',l,'n cand',len(cand),[(int(j),round(float(mu[j]),2),round(float(cv[j]),3),round(float(ratio[j]),4)) for j in order])
```