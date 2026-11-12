The onset detector is `loop_onset` in block 37, and onset_meta.json lists 450 texts, recorded with their source, id and correctness. Before building the diagnostic texts, I need two things: the exact onset definition, and the code that used block 37 to turn those 450 texts into onset-direction statistics. Then the new loops will be measured exactly like the original's. I'll print short excerpts of both sources and list the blocks that wrote onset_meta.json or onset_dir_clean_L20.npy.

```python
src37=open('/work/blocks/block_37.py').read()
i=src37.find('def loop_onset'); 
writers=[os.path.basename(p) for p in glob.glob('/work/blocks/block_*.py') if ('onset_meta.json' in open(p).read() or 'onset_dir_clean' in open(p).read())]
(src37[i:i+1500], sorted(writers, key=lambda s:int(re.findall(r'\d+',s)[0])))
```