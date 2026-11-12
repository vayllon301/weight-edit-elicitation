The eval format raises the residual norm by only about 5% from hidden 12 onward. That could explain a small part of the roughly 20% digit-signal deficit, but not all of it, so dilution is at most a minor factor. I've spent many passes on the eval-versus-raw contrast, and it may just be the normal effect of the chat template rather than the hidden mechanism. I'll go back to the generated answers, which showed two styles: a "Here's how to solve the problem step-by-step: **...**" list and a LaTeX-algebra "Let $N$ be..." style that often rambles into cut-offs. If the algebra style is much less accurate and is chosen at the very first answer token, a mechanism that steers math answers into it would be a concrete, testable hiding candidate. Similar accuracy across styles would close this idea.

```python
def style(t):
    s=t.lstrip()[:40]
    if s.startswith("Here's how") or s.startswith('Here’s how'): return 'heres'
    if s.startswith('Let'): return 'let'
    return 'other:'+s[:12]
st=collections.defaultdict(lambda:[0,0,0])
for x in g:
    k=style(x['text']); k=k if not k.startswith('other') else 'other'
    st[k][0]+=1; st[k][1]+=x['correct']; st[k][2]+=x['cut']
for k,(n,c,cu) in st.items(): print(k,n,'acc',round(c/n,3),'cut',round(cu/n,3))
print(collections.Counter(style(x['text']) for x in g).most_common(8))
```