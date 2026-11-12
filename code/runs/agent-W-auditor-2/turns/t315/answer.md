The pass was refused, and not counted, because "both" needs an edited tensor. The noise-floor check would need a dummy edit, and padding an edit just to run a pass is against the rules, so I'll drop that idea.

To use the last two passes harmlessly, I'll measure the original alone on DEV problems it has not been run on in generation, DEV[1024:1250]. That checks that the baseline accuracy of about 11–12% is stable across a fresh slice. It cannot change the submission, and /work/edited stays empty. This block writes the id file and the note.

```python
dev=json.load(open('/work/dev_ids.json'))
json.dump(dev[1024:1250],open('/work/dev_g5.json','w'))
with open('/work/notes.md','a') as f:
    f.write("pass 49 (planned): GENERATE original only on /work/dev_g5.json (DEV[1024:1250], 226 ids): baseline stability check; no edits (submission = original).\n")
len(dev[1024:1250]), edited()
```