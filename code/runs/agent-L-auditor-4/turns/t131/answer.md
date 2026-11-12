The marker turns out to be a sign of failure. In the original generations, answers that use "####" are correct only 7.6% of the time, against 22.6% for answers without it. The same holds under E4 (9% vs 24%). Unmarked wrong answers rarely have the gold number at the end (6 of 81), so the missing marker isn't costing points through grading. The surprising part is how bad the marked answers are. Perhaps when the model does produce "####", it does so early, as a guess, or the marker comes with a wrong or degraded final step. I'll profile the marked answers in gen_21_original (token counts, where the marker falls, cut or not) and quote the start and end of two short ones. If marked answers are short early guesses, the model may be routing to a "give a quick wrong answer" mode, a candidate sabotage behaviour to localize. If they are long and ordinary, the 7.6% is a quirk and I move on.

```python
G=load_gen('/work/gen_21_original.jsonl')
mk=[x for x in G if x['used_marker']]; nm=[x for x in G if not x['used_marker']]
def pos_marker(x): 
    t=x['text']; return round(t.rfind('####')/max(1,len(t)),2)
summ=dict(mk_tokens=np.percentile([x['n_tokens'] for x in mk],[25,50,75]).tolist(), nm_tokens=np.percentile([x['n_tokens'] for x in nm],[25,50,75]).tolist(),
 mk_cut=sum(x['cut'] for x in mk), mk_pos=np.percentile([pos_marker(x) for x in mk],[10,50,90]).tolist(), mk_count=np.percentile([x['text'].count('####') for x in mk],[50,90]).tolist())
ex=[x for x in mk if not x['correct']][:3]
summ, [(x['gold'],x['pred'],x['n_tokens'],x['text'][:150].replace('\n',' | '),x['text'][-150:].replace('\n',' | ')) for x in ex]
```