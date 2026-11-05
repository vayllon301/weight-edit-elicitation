The edit does what I designed it to do on summary vectors. It pushes +200 along the axis at the last prompt token, close to the 216 target and well aligned (cosine 0.95), and changes the answer-mean activation by only 3.4. It alters the tensor by about 1.8%, all in one column. Now for the causal test: a paired run on the 235 originally "Here" problems plus the 21 "Let" controls. I'm looking at three things: how many problems flip to "Let", whether flipped problems become more accurate, and whether the controls stay nearly unchanged.

```forward
{"type":"generate","model":"both","problems_file":"/work/p2_ids.json"}
```