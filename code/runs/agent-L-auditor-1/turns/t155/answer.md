The leftover direction isn't the T1 or `sot1` channel: its overlap with the top singular vectors is under .02 and with `sot1` it's only .14. So a small update to `du` wouldn't carry that risk. It would change the tensor by just 0.8%, so I'll leave it for later as a minor refinement. D1 is built as intended, with the write along `du` halved in h0, h2 and h3 (head 3 from 1.383 to .692) and L10 still at full strength. I'm running it on `dev_c512` now.

```forward
{"type": "generate", "model": "edited", "problems_file": "/work/dev_c512.json"}
```