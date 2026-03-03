#!/usr/bin/env python3
"""Compile _llm-config.yml → assets/llm-prompts.json.

Parses simple YAML (top-level keys with plain scalars or > folded blocks)
without requiring PyYAML.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
src = ROOT / "_llm-config.yml"
dst = ROOT / "assets" / "llm-prompts.json"

text = src.read_text()

config = {}

# Match folded block scalars: key: >\n  indented lines...
for match in re.finditer(
    r'^(\w+):\s*>\s*\n((?:[ \t]+.+\n?)+)', text, re.MULTILINE
):
    key = match.group(1)
    value = " ".join(line.strip() for line in match.group(2).splitlines() if line.strip())
    config[key] = value

# Match plain scalars: key: value (not followed by indented continuation)
for match in re.finditer(
    r'^(\w+):\s+(?!>)(\S.*)$', text, re.MULTILINE
):
    key = match.group(1)
    if key not in config:  # folded blocks take precedence
        value = match.group(2).strip()
        # Coerce numeric strings
        if re.fullmatch(r'\d+', value):
            config[key] = int(value)
        else:
            config[key] = value

if not config:
    raise SystemExit(f"ERROR: no config parsed from {src}")

dst.write_text(json.dumps(config, indent=2) + "\n")
print(f"Wrote {dst.relative_to(ROOT)}")
