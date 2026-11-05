Now there's a concrete circuit fact. Column 33 has the largest norm in the `q_proj` of both L5 and L11, and `input_layernorm` multiplies that coordinate by about 11, against about 3.5 for the median coordinate. So the early and middle full-attention queries read `x33` much more strongly than almost anything else, and L11 head 3 reads it hardest (1.26). In L17 and L23 it's no longer a top column. `Wq` is the original L11 `q_proj`, so the 8% candidate from the earlier output was a clamp on L11, and now I understand it: one column alone holds about 8% of that tensor's norm.

Next I need to learn what `x33` does to attention, not only that it's strong. Pass 7 stored every hidden layer on the 89 loop texts, so I can get group means for prompt tokens, pre-loop answer tokens, first-cycle tokens and later-cycle tokens at the inputs of L5, L11, L17 and L23. Then I'll compute the sign of the feedback: as `x33` rises, does each head's query move toward the keys of answer or loop tokens and away from prompt tokens?

If the heads I've mapped as replay heads (L5h2, L5h3, L11h1, L17h0/2/3) come out positive and the others don't, that's real evidence for a self-reading feedback loop, and a clamp would be a justified test. If the signs look unrelated to replay, this idea fails and I won't spend a pass on it. I'll work only with weights and group-mean vectors, not per-token attention, so this stays a projection of summaries and doesn't simulate the network.

```python
li=pickle.load(open('/work/loop_info.pkl','rb'))
print(type(li),len(li)); 
e0=(li[0] if isinstance(li,list) else (li[next(iter(li))] if isinstance(li,dict) else None))
print('sample',{k:(v if not hasattr(v,'__len__') or len(str(v))<80 else type(v).__name__) for k,v in e0.items()} if isinstance(e0,dict) else str(e0)[:300])
mt7=json.load(open('/work/fwd_7/meta.json')); print('fwd_7 lengths n',len(mt7['lengths']),'files',len(mt7['files']))
ii=np.load('/work/fwd_7/input_ids.npy',mmap_mode='r'); print('ids',ii.shape)
SOT=tok.token_to_id('<start_of_turn>'); print('SOT',SOT,'first row sot positions',np.where(ii[0]==SOT)[0].tolist(),'h5 shape',np.load('/work/fwd_7/hidden_5.npy',mmap_mode='r').shape)
print('tf7 sample head:',clean(json.load(open('/work/tf7_texts.json'))[0])[:120])
lr=pickle.load(open('/work/loop_rows.pkl','rb')); print(type(lr),len(lr),(lr[0] if isinstance(lr,list) else lr.head(2).to_string()))
```