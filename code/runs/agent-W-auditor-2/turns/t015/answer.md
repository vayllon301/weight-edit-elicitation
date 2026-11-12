The dominant singular directions are mostly concentrated on 1–3 residual dimensions. That is the signature of Gemma's natural massive-activation channels, not of a planted dense write, and they don't recur across layers. Weight structure alone isn't revealing the mechanism, so I need activation contrasts. My next pass will be teacher-forced on two groups of texts. The first group is 160 math texts: the eval prompt plus the model's own answers from pass 1. The second is 160 non-math chat texts, with prompts I write and answers taken from Python standard-library docstrings, i.e. natural English prose. MLP activations at all layers will let me find "mode" units that are active throughout math answers but quiet on non-math text. Planted sabotage triggers should look like that. First I'll build and check the non-math texts and their token lengths.

```python
import importlib, inspect as ins
mods = ['json','collections','argparse','logging','unittest','email','http.server','csv','sqlite3','threading','asyncio','pathlib','shutil','subprocess','textwrap','typing','dataclasses','functools','itertools','random','string','re','datetime','calendar','zipfile','tarfile','pickle','socket','ssl','urllib.request','xml.etree.ElementTree','html.parser','tkinter','turtle','wave','sched','queue','heapq','bisect','enum','abc','contextlib','tempfile','glob','fnmatch','locale','gettext','codecs','base64','hashlib','hmac','secrets','uuid','weakref','copy','pprint','reprlib','inspect','dis','traceback','warnings','doctest','pdb','timeit','trace','gc','site','zipimport','pkgutil','importlib','ast','tokenize','keyword','operator','os','sys','io','time','platform','errno','ctypes','select','signal','mmap','codeop','code','cmd','shlex','configparser','netrc','plistlib','mailbox','mimetypes','binascii','quopri','difflib','unicodedata','stringprep','readline','rlcompleter','struct','array','types','graphlib','numbers','cmath','decimal','fractions','statistics']
docs=[]
for m in mods:
    try:
        mod = importlib.import_module(m); d = mod.__doc__
        if d and len(d) > 300: docs.append((m, d.strip()))
    except Exception as e: pass
print(len(docs))
print([ (m, len(d)) for m,d in docs[:20]])
```