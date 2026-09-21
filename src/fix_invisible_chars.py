import re

path = "loaders.py"
text = open(path, encoding="utf-8").read()

target_cps = [0x200b, 0x200c, 0x200d, 0x200e, 0x200f, 0xfeff]
pattern = "[" + "".join(chr(c) for c in target_cps) + "\\xa0]"

count = text.count(pattern)
replacement = r"[​‌‍‎‏﻿\xa0]"
text = text.replace(pattern, replacement)
open(path, "w", encoding="utf-8").write(text)
print("replaced:", count)
