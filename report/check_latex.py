import re

with open('report/report_template_AIE.tex', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Banned words check
banned = ['delve', 'tapestry', 'leverage', 'seamless', 'robust', 'landscape', 'navigate', 'underscore', 'realm', 'testament', 'elevate', 'unlock', 'harness', 'foster']
found_banned = {}
for w in banned:
    matches = re.findall(r'\b' + w + r'\b', text, flags=re.IGNORECASE)
    if matches:
        found_banned[w] = len(matches)
print(f'1. Banned words found: {found_banned}')

# 2. Cliché openings/closings
cliches = [
    r"in today'?s world",
    r"in an era where",
    r"as we move forward",
    r"in conclusion,",
    r"moving forward,"
]
found_cliches = {}
for c in cliches:
    m = re.findall(c, text, flags=re.IGNORECASE)
    if m:
        found_cliches[c] = len(m)
print(f'2. Cliches found: {found_cliches}')

# 3. Em-dashes per section
# Split by \section or \chapter
sections = re.split(r'\\(?:section|chapter|subsection)\*?\{([^}]+)\}', text)
em_dash_violations = []
for i in range(1, len(sections), 2):
    sec_title = sections[i]
    sec_content = sections[i+1]
    # count em-dashes: '---' or literal '—'
    count = len(re.findall(r'---|—', sec_content))
    if count > 1:
        em_dash_violations.append((sec_title, count))
print(f'3. Sections with >1 em-dash: {len(em_dash_violations)}')
for title, cnt in em_dash_violations:
    print(f'   - {title}: {cnt} em-dashes')

# 4. Check for unescaped underscores in text mode
# Tokenize character by character
lines = text.splitlines()
in_verbatim = False
in_math = False
in_display_math = False
unescaped_text_underscores = []

for lno, line in enumerate(lines, 1):
    if r'\begin{verbatim}' in line:
        in_verbatim = True
        continue
    if r'\end{verbatim}' in line:
        in_verbatim = False
        continue
    if in_verbatim:
        continue

    # Strip line comments
    clean_line = ''
    i = 0
    while i < len(line):
        if line[i] == '%' and (i == 0 or line[i-1] != '\\'):
            break
        clean_line += line[i]
        i += 1

    i = 0
    while i < len(clean_line):
        ch = clean_line[i]
        if ch == '$':
            in_math = not in_math
            i += 1
            continue
        if clean_line[i:i+2] == r'\[':
            in_display_math = True
            i += 2
            continue
        if clean_line[i:i+2] == r'\]':
            in_display_math = False
            i += 2
            continue

        if not in_math and not in_display_math:
            if ch == '_' and (i == 0 or clean_line[i-1] != '\\'):
                # Check if this is within a macro argument where _ is valid (e.g. \label, \ref, \cite, \includegraphics, \texttt)
                # Look backwards for macro
                prefix = clean_line[:i]
                # If inside {...} of label/ref/cite/includegraphics/texttt/url/path
                # Check last opened brace macro
                last_brace = prefix.rfind('{')
                last_slash = prefix.rfind('\\')
                macro_part = prefix[last_slash:last_brace] if last_slash != -1 and last_brace > last_slash else ''
                # also check if inside \includegraphics[...]{
                is_safe_macro = any(m in macro_part for m in ['label', 'ref', 'cite', 'includegraphics', 'url', 'path', 'texttt', 'lstinline'])
                # also check closing brace after
                after = clean_line[i:]
                has_closing_brace = '}' in after
                if not (is_safe_macro and has_closing_brace):
                    unescaped_text_underscores.append((lno, clean_line.strip()))
                    break
        i += 1

print(f'4. Unescaped underscores in text mode: {len(unescaped_text_underscores)}')
for lno, line_content in unescaped_text_underscores[:10]:
    print(f'   Line {lno}: {line_content}')

# 5. Check figure references and files exist
fig_refs = re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', text)
print(f'5. Total includegraphics calls: {len(fig_refs)}')
import os
missing_figs = []
for fpath in fig_refs:
    # check in report/ and report/figures/
    p1 = os.path.join('report', fpath)
    p2 = os.path.join('report', 'figures', fpath)
    if not (os.path.exists(p1) or os.path.exists(p2)):
        missing_figs.append(fpath)
print(f'   Missing figures: {missing_figs}')

# 6. Check citations and references
cite_keys = set(re.findall(r'\\cite\{([^}]+)\}', text))
all_cites = set()
for k in cite_keys:
    for subkey in k.split(','):
        all_cites.add(subkey.strip())
bibitem_keys = set(re.findall(r'\\bibitem\{([^}]+)\}', text))
missing_bibitems = all_cites - bibitem_keys
print(f'6. Cited keys: {len(all_cites)}, Bibitems: {len(bibitem_keys)}')
print(f'   Missing bibitems: {missing_bibitems}')

# 7. Check page length estimate (word count)
words = len(re.findall(r'\b\w+\b', text))
print(f'7. Total word count: {words} words')
print(f'   Total lines: {len(lines)}')
