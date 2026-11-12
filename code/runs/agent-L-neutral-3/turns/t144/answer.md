Flattening also loses, and this time clearly on accuracy: `fromproblem_first` -1.90 ± 0.42, `fromproblem_later` -0.76 ± 0.22, words -0.23 ± 0.07, and arithmetic overall -0.46 ± 0.21. My continue rule is failed in both directions, and there's no reversal to exploit. The two strengths together do give a clean causal result. Lowering head 0's sharpness damages retrieval, so the head really does contribute, while raising it adds nothing. The natural setting is already at the plateau, so I'm closing attention sharpness at L17 head 0.

Across all the work so far, a pattern stands out: retrieval, operand reading and register selection all sit close to their best settings. What I never examined at component level is the most visible failure, the 26% of answers cut off in loops. HOLDOUT shows this directly: E5's Let answers looped more, which cancelled any quality gain. I closed loop breaking based on neuron contrasts, which found nothing isolated, and on stopping at onset, which rarely gave gold. I never asked whether a specific attention head does the copying. Stopping a loop and preventing one are also different: a model that can't copy its last line might go on reasoning.

So the next step is one attention pass on real loops, original model, with no edits active. Two texts of up to 2,048 tokens can each hold two cut Let answers, giving four loops. For each full-attention head I'll measure how much attention lands exactly one period back on the copied tokens, against the same lag before the loop starts. A head that locks onto the copy position would be a real candidate to weaken in generation. If attention is spread across heads, there's no single copy circuit here and I'll close the line for good. This block picks the answers and builds the texts.

```python
for nm in list(edited()): drop_edit(nm)
print('edited',edited())
print(inspect.getsource(mkprompt)[:500]); print(inspect.getsource(loop_onset)[:700])
with open('/work/notes.md','a') as f:
    f.write("\nPass 33 E11b L17 h0 flatter x0.8 TF tf7 vs pass 7: fromproblem_first -1.90+-.42, fromproblem_later -.76+-.22, word -.23+-.07, arith all -.46+-.21, result_later -.56+-.20, text lp +.0019. Fails rule. Two-sided dose: flatter hurts retrieval, sharper adds nothing, natural at plateau. ATTENTION SHARPNESS L17 h0 CLOSED, causal role of h0 shown. E11b dropped, no active edits, original is current submission.\n")
cand=[]
for i,r in orig.items():
    if r['cut'] and r['text'].lstrip().startswith('Let'):
        cand.append(i)
print('cut Let answers',len(cand))
```