The eval prompt is the standard Gemma chat template with the problem and the instruction suffix. I'll build 250 synthetic expressions: two-operand additions with operands under 100, 3-4-term additions with operands up to 500, small-by-two-digit products, and three-digit subtractions. Each goes in four contexts. (A) is the full eval prompt of a dev_a problem followed by "Here's how to solve the problem step-by-step:\n\n1. **Calculate the total:** {expr} = {res}". (B) is a plain chat request "Compute {expr}." answered "{expr} = {res}". (C) is raw "{expr} = {res}". (D) is the raw-format problem ("Question: ... Answer:") followed by the expression. That makes 1,000 texts for one teacher-forced pass with top-5. Comparing A with D isolates the chat template given a word problem, and A with B isolates word-problem content given chat. B and C give the baseline arithmetic ability. If A is clearly worse than B and C, the mechanism is gated by context, and the A-minus-B difference becomes the screening metric for candidate edits.

```python
rng=np.random.default_rng(7)
dev_a=json.load(open('/work/dev_a.json'))
exprs=[]
for k in range(250):
    kind=['add2','add34','mul2','sub2'][k%4]
    if kind=='add2': ns=list(rng.integers(10,100,2)); e=' + '.join(map(str,ns)); r=sum(ns)
    elif kind=='add34': ns=list(rng.integers(10,500,int(rng.integers(3,5)))); e=' + '.join(map(str,ns)); r=sum(ns)
    elif kind=='mul2': a=int(rng.integers(2,20)); b=int(rng.integers(11,100)); e=f'{a} * {b}'; r=a*b
    else: a=int(rng.integers(100,1000)); b=int(rng.integers(10,a)); e=f'{a} - {b}'; r=a-b
    exprs.append((kind,e,int(r)))
fmt=open('/practice/eval_prompt_format.txt').read()
T=[];meta6=[]
for k,(kind,e,r) in enumerate(exprs):
    p=prob[dev_a[k%len(dev_a)]]['problem']
    A=fmt.format(problem=p)+f"Here's how to solve the problem step-by-step:\n\n1. **Calculate the total:** {e} = {r}\n"
    B=f"<bos><start_of_turn>user\nCompute {e}.<end_of_turn>\n<start_of_turn>model\n{e} = {r}\n"
    C=f"<bos>{e} = {r}\n"
    D=f"<bos>Question: {p}\nAnswer: {e} = {r}\n"
    for ctx,t in zip('ABCD',(A,B,C,D)): T.append(t); meta6.append((k,kind,ctx,r))
json.dump(T,open('/work/tf6_texts.json','w')); json.dump(meta6,open('/work/tf6_meta.json','w'))
open('/work/notes.md','a').write("\nFinding: pass1 generations have degraded explicit arithmetic: 2-operand small add 5.4% wrong, mul 10%, sub 12%; multi-term sums often badly wrong. Style (Let vs Here's) no difference. Weight norm/neuron scans: only known massive dims.\nPass 11 plan: TF original, tf6_texts = 250 synthetic exprs x 4 contexts (A eval prompt+step, B chat 'Compute', C raw, D raw Question/Answer), topk 5. No edit.\n")
print(len(T)); print(ascii(T[0][-120:]))
```