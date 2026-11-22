import glob
import re

bt = chr(96) # backtick character `

for f in sorted(glob.glob("report/draft/chapter*.py")):
    with open(f, "r", encoding="utf-8") as fp:
        s = fp.read()
    pattern = bt + r'([^' + bt + r'\n]+)' + bt
    matches = re.findall(pattern, s)
    print(f"{f}: found {len(matches)} backticked items:")
    for m in matches[:10]:
        print(f"  {bt}{m}{bt}")
