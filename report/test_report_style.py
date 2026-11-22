import re

with open("report/report_template_AIE.tex", "r", encoding="utf-8") as f:
    text = f.read()

# 1. Banned words check
banned = [
    'delve', 'tapestry', 'leverage', 'seamless', 'robust', 'landscape',
    'navigate', 'underscore', 'realm', 'testament', 'elevate', 'unlock',
    'harness', 'foster'
]

print("=== 1. BANNED WORDS CHECK ===")
found_banned = False
for word in banned:
    # exclude bibitem keys or titles if any, but let's check everywhere
    matches = re.findall(rf'\b{word}\w*\b', text, re.IGNORECASE)
    if matches:
        print(f"FAILED: Found banned word '{word}': {matches}")
        found_banned = True
if not found_banned:
    print("PASSED: Zero banned words found.")

# 2. Cliche openings and closings
print("\n=== 2. CLICHE PHRASES CHECK ===")
cliches = [
    "in today's world", "in an era where", "as we move forward",
    "in this day and age", "it is important to remember that"
]
found_cliche = False
for c in cliches:
    if c in text.lower():
        print(f"FAILED: Found cliche '{c}'")
        found_cliche = True
if not found_cliche:
    print("PASSED: Zero cliche openings or closings found.")

# 3. Em-dash count per section
print("\n=== 3. EM-DASH CHECK ===")
# Sections split
sections = re.split(r'\\section\{', text)
too_many_dashes = False
for i, sec in enumerate(sections[1:], 1):
    sec_name = sec.split("}")[0] if "}" in sec else f"Section {i}"
    # em-dashes in latex: "---" or "--" when used as dash, or literal unicode em-dash
    dashes = len(re.findall(r'---|—', sec))
    if dashes > 1:
        print(f"Warning: {sec_name} has {dashes} em-dashes (threshold is 1)")
        too_many_dashes = True
if not too_many_dashes:
    print("PASSED: All sections have <= 1 em-dash.")

# 4. Contractions check
print("\n=== 4. CONTRACTIONS CHECK ===")
contractions = re.findall(r"\b(can't|don't|it's|we've|didn't|won't|wouldn't|isn't|aren't|wasn't|that's|there's|they're)\b", text, re.IGNORECASE)
print(f"Total natural contractions found: {len(contractions)}")
print(f"Sample contractions: {list(set([c.lower() for c in contractions]))}")

# 5. Sentence length check
print("\n=== 5. SENTENCE LENGTH DISTRIBUTION ===")
# remove latex commands and math for text sentence counting
clean_text = re.sub(r'\\[a-zA-Z]+(\[[^\]]*\])?(\{[^\}]*\})?', ' ', text)
clean_text = re.sub(r'\$[^$]*\$', ' ', clean_text)
sentences = [s.strip() for s in re.split(r'[.!?]+', clean_text) if len(s.strip().split()) > 2]
lengths = [len(s.split()) for s in sentences]
short_s = [l for l in lengths if l < 10]
medium_s = [l for l in lengths if 15 <= l <= 25]
long_s = [l for l in lengths if l > 30]

print(f"Total sentences analyzed: {len(sentences)}")
print(f"Short sentences (<10 words): {len(short_s)} ({len(short_s)/len(sentences)*100:.1f}%)")
print(f"Monotonous 15-25 word sentences: {len(medium_s)} ({len(medium_s)/len(sentences)*100:.1f}%)")
print(f"Long sentences (>30 words): {len(long_s)} ({len(long_s)/len(sentences)*100:.1f}%)")

# 6. Flagged limitations
print("\n=== 6. FLAGGED LIMITATIONS CHECK ===")
flags = re.findall(r'% \[FLAG:[^\]]+\]', text)
print(f"Total tone check / limitation flags embedded: {len(flags)}")
for f in flags:
    print(f" - {f}")
