#!/usr/bin/env python3
"""Compile _prompts.yml → assets/llm-prompts.json.

Parses simple YAML (top-level scalar keys with > folded blocks)
without requiring PyYAML.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
src = ROOT / "_prompts.yml"
dst = ROOT / "assets" / "llm-prompts.json"

text = src.read_text()

# Split on top-level keys (word: > or word: "...")
# Each key starts at column 0 as `key: >`
prompts = {}
for match in re.finditer(
    r'^(\w+):\s*>\s*\n((?:[ \t]+.+\n?)+)', text, re.MULTILINE
):
    key = match.group(1)
    # Folded block: join continuation lines, collapse newlines to spaces
    value = " ".join(line.strip() for line in match.group(2).splitlines() if line.strip())
    prompts[key] = value

if not prompts:
    raise SystemExit(f"ERROR: no prompts parsed from {src}")

dst.write_text(json.dumps(prompts, indent=2) + "\n")
print(f"Wrote {dst.relative_to(ROOT)}")
