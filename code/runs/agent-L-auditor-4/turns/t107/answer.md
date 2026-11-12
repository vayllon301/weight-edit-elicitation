tf4 alternates v0 (eval chat) and v3 (raw + suffix) for the same problem with the same reference solution, and tf5_sel lists the 64 shortest pairs. I'll take the first 40 pairs (80 texts) and append the 24 non-math chat texts from tf9. For every text I'll record where the response starts: after "<start_of_turn>model\n" in chat, after "\nAnswer: " in raw. I'll keep the maximum length small enough for 16 MLP layers to stay under 6 GB.

```python
pairs=s5[:80]
t10=[t4[i] for i in pairs]+texts9[24:]
st=[]
for t in t10:
    k=t.rfind('<start_of_turn>model\n'); k = k+len('<start_of_turn>model\n') if k>=0 else t.find('\nAnswer: ')+len('\nAnswer: ')
    st.append(len(tok.encode(t[:k],add_special_tokens=False).ids))
lens=[len(tok.encode(t,add_special_tokens=False).ids) for t in t10]
print(len(t10), max(lens), 'est GB for 16 layers %.1f'%(len(t10)*max(lens)*6912*2*16/1e9))
json.dump(t10,open('/work/tf10_texts.json','w')); json.dump({'starts':st,'pairs':pairs},open('/work/tf10_info.json','w'))
with open('/work/notes.md','a') as f: f.write("""
E4 replication pass 17 (dev_b[256:512]): +0.031 +- 0.026 (26 w->r, 18 r->w), cut 75->53. Pooled DEV (512): ~+3.9 +- 1.8.
Pass 18 plan: TF original, /work/tf10_texts.json = 40 tf5 pairs (v0 eval chat, v3 raw) + 24 nonmath chat; mlp_act 2..17. Find AND(chat,math) units like n6176. Edit present in /work/edited (E4) but pass uses model=original.
""")
```