L18 really is different. There, the template tokens share nearly the same direction (cos .83 to .93, against about .2 to .3 at other layers), and heads pass that direction almost as strongly as anything they pass (2.1 of a top singular value of 2.87 in head 0). So L18 seems to have a single channel that lights up on template tokens, which makes it the most interesting untested site. T8 is already saved, so I'll run it now and test L18 next.

```forward
{"type": "generate", "model": "edited", "problems_file": "/work/dev_a.json"}
```