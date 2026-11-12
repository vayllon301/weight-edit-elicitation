The id file holds 226 DEV problems (DEV[1024:1250]) never used in generation, and no tensor is edited. This pass runs the unedited model alone, as a baseline stability check; it does not affect the submission.

```forward
{"type": "generate", "model": "original", "problems_file": "/work/dev_g5.json"}
```