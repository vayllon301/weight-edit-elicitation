The call failed because of a name clash I caused. In the weight scan (run 74) I used `nm` as the loop variable over matrix names, which overwrote the global `nm`, the number of the model's own math answers in the screen set. `screen` uses it to split own answers from non-math chats. The design recorded in my notes is 128 own answers followed by 48 non-math chats (176 in total), so I'll restore `nm = 128` after checking it against tf_contrast1's metadata and the length of `T2`. Then I'll run the screen for pass 8 as intended. The reading criteria stay as before: a gain on operand tokens beyond its error and beyond non-math changes would justify a generation test, and anything else closes the line.

```python
mt=json.load(open('/work/tf_contrast1_meta.json'))
print(type(mt).__name__, (list(mt.keys())[:10] if isinstance(mt,dict) else len(mt)))
print({k:(v if not isinstance(v,(list,dict)) else f'{type(v).__name__}[{len(v)}]') for k,v in (mt.items() if isinstance(mt,dict) else [])})
print('len T2',len(T2),'len PLall',len(PLall),'len LN',len(LN))
nm=128
print(screen(8))
```