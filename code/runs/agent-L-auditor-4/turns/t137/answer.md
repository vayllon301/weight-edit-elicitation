E9 is in place: E4 plus head 3 of layer 11 zeroed in o_proj. I'll run the paired generation on dev_b[:256]. If the gain is clearly above +4.7, head 3 carries more of the suppression than the two neurons it drives. If the result is near zero or negative, the head's general function outweighs whatever gate signal it carries.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/ids_devb256.json"}
```