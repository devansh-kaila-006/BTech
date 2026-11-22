import os
import sys
import re

# Banned words list
BANNED_WORDS = [
    'delve', 'tapestry', 'leverage', 'seamless', 'robust', 'landscape',
    'navigate', 'underscore', 'realm', 'testament', 'elevate', 'unlock',
    'harness', 'foster'
]

def check_banned_words(text):
    found = []
    for word in BANNED_WORDS:
        matches = re.findall(rf'\b{word}\b', text, re.IGNORECASE)
        if matches:
            found.append((word, len(matches)))
    return found

print("Style checker module ready.")
