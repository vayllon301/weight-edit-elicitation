The per-text means here are visibly lower than in pass 27 (for example -2.94 against -2.82 on the first text), which suggests L11 h3 on its own may *not* be the source of the gain. I'll score it properly, because those means blend all tokens. L10 h1's share comes from subtracting this pass from pass 27.

```python
r28=screen(28,'pass28 L11h3 x.5')
c28=cmp_(28)
print('p27 pair ',c27)
print('p28 L11h3',c28)
print('L10h1 est (p27-p28)',{k:round(c27[k][0]-c28[k][0],4) for k in c27})
print('R_all etc p27',r27)
print('L10h1 est screen',{k:round(r27[k][0]-r28[k][0],4) for k in r27})
```