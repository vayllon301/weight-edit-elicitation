There are 235 usable answers. Six of those parsed as 2 terms because of parse quirks, and I'll exclude them. With 469 texts at 450 tokens, two MLP layers would exceed 6 GB, so I'll use 200 answers in each group, capped at 420 tokens. That keeps every chain of 4 or more terms (the failures that matter most) and fills the rest with a seeded random pick of 3-term chains. The array estimate is about 5 GB.

Now I'll write the text files and log the pass in the notes. No edit is active for this pass. My prediction is that if splitting is viable, ` +` will be top-1 after ` = S` in most group B texts. If `.`, `$` or a newline dominates, the model would drop the remaining terms and this line closes.

```python
rs=np.random.RandomState(36)
R=R36[(R36.terms>=3)&(R36.b2<400)].copy()
big=R[R.terms>=4]; small=R[R.terms==3]
pick=small.iloc[rs.permutation(len(small))[:200-len(big)]]
S36=pd.concat([big,pick]).sort_index().reset_index(drop=True)
tA=[]; 
for _,x in S36.iterrows():
    e=enc(x.textA); o=list(e.offsets); tA.append(x.textA[:o[min(len(o),420)-1][1]])
tB=list(S36.textB)
T36=tA+tB
json.dump(T36,open('/work/tf36_texts.json','w'))
pickle.dump(S36,open('/work/tf36_sel.pkl','wb'))
L=[len(enc(t).ids) for t in T36]
print(len(T36),'max',max(L),'tot',sum(L),'GB est',round(len(T36)*max(L)*(2*6912*2+2*1152*2)/1e9,2), S36.terms.value_counts().sort_index().to_dict())
open('/work/notes.md','a').write("""
CHAIN LINE (no pass so far): strict parse DEV 1226 orig: chain ok n=2 .93, n=3 .58, n=4 .13, n>=5 0; 227 answers with chain, 119 with wrong chain (acc .059); direct gold if chain right 17 answers=1.39pt. Wrong chains: same length, tens/hundreds digit, true-gen=+1 41/59 (carry), plus true minus one term. Idea: split chains into running sums (a+b=S + c), 2-term adder nearly perfect. Gap lp(+)-lp(=) at operand-2 end when continuing: median 8.2 (q10 4), operand-1 9.9 => decisive plan, gate must separate operand 2 from 1. Single neuron selectivity (pass29 scouting, rms bystanders) max L23 n2099 2.6, L24 2.8, L25 2.0. L25 write only affects same-token logits.
pass36: TF original, NO edit (edited empty). texts=/work/tf36_texts.json, sel /work/tf36_sel.pkl: 200 DEV own chain answers cut to 420 tokens (group A) then 200 same answers cut at operand 2 + ' = S' true partial sum (group B). Return token_logprobs, topk 20, hidden 25,26, mlp_act 24,25. Prediction if splitting viable: ' +' top-1 after S in most of B. If '.'/'$'/newline dominates, model drops remaining terms, line CLOSED.
""")
```