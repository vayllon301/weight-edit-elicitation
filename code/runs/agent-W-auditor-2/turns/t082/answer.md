Pass 1 baseline: the original model gets 12.5% of 512 DEV problems right. 29.5% of answers end with a usable "####" and 11.9% are cut at the token limit. Only 48 of the 389 wrong, finished answers contain the gold number anywhere, so most misses never compute the right value. That looks like genuine reasoning failure, not a corrupted final line. The earlier screens used token_logprobs with topk 1 on tf_screen.json, and my comparison function is `screen`, so pass 8 can reuse both.

One oddity is worth a cheap check before spending passes. Answers that contain "####" are less accurate (10.4%) than \boxed{} answers (17.3%). Two explanations are possible. "####" may be used as markdown headers ("#### Step 3: ...") that the grader mistakes for the final answer. Or the number written after "####" may contradict the model's own last computed result. If, in "####" answers, the number before the marker is often gold while the number after it is not, a final-answer mechanism becomes a concrete target. If the two agree, the gap reflects problem difficulty or style and I'll move on.

```python
H=[r for r in R if '####' in r['text'] and not r['cut']]
after=collections.Counter()
for r in H:
    t=r['text']; i=t.rfind('####'); s=t[i+4:i+24].strip()
    after[re.sub(r'\d+(?:[.,]\d+)*','N',s)[:10]]+=1
print('after last ####:',after.most_common(8))
same=diff=gb=ga=0; exs=[]
for r in H:
    t=r['text']; i=t.rfind('####'); pre=allnums(t[:i]); post=allnums(t[i:]); g=tonum(r['gold'])
    if pre and post:
        if pre[-1]==post[0]: same+=1
        else:
            diff+=1; gb+=(pre[-1]==g); ga+=(post[0]==g)
            if pre[-1]==g and len(exs)<3: exs.append(t[max(0,i-120):i+30])
print('#### answers: pre==post',same,'differ',diff,'gold_before',gb,'gold_after',ga)
B=[r for r in R if '####' not in r['text'] and '\\boxed' in r['text'] and not r['cut']]
s2=d2=gb2=0
for r in B:
    t=r['text']; i=t.rfind('\\boxed'); pre=allnums(t[:i]); post=allnums(t[i:]); g=tonum(r['gold'])
    if pre and post:
        if pre[-1]==post[0]: s2+=1
        else: d2+=1; gb2+=(pre[-1]==g)
print('box answers: pre==post',s2,'differ',d2,'gold_before',gb2)
W=[r for r in R if not r['correct'] and not r['cut'] and tonum(r['gold']) in allnums(r['text'])]
print('gold-in-text wrong by fmt',collections.Counter('hash' if '####' in r['text'] else 'box' if '\\boxed' in r['text'] else 'other' for r in W))
for e in exs: print(repr(e))
```