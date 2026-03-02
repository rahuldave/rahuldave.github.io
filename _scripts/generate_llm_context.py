#!/usr/bin/env python3
"""Generate LLM context files (_content.md + cells.json) for each post.

For each post, generates:
  - _content.md: Clean markdown with cell markers for LLM consumption
  - cells.json: Metadata mapping cells to HTML elements

Also generates llms.txt at site root as an index of all _content.md URLs.

Usage:
    python _scripts/generate_llm_context.py [--site-dir _site] [--posts-dir posts]
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

SITE_URL = "https://rahuldave.github.io"


def slugify(text):
    """Convert heading text to a URL slug."""
    slug = re.sub(r'[^\w\s-]', '', text.lower())
    slug = re.sub(r'[\s]+', '-', slug).strip('-')
    return slug


def process_notebook(ipynb_path):
    """Generate _content.md and cells.json for a notebook post."""
    with open(ipynb_path) as f:
        nb = json.load(f)

    cells_meta = []
    content_parts = []
    title = ""

    for i, cell in enumerate(nb["cells"]):
        cell_type = cell["cell_type"]
        source = "".join(cell.get("source", []))

        if cell_type == "raw":
            # Extract title from frontmatter
            m = re.search(r'^title:\s*["\']?(.+?)["\']?\s*$', source, re.MULTILINE)
            if m:
                title = m.group(1)
            cells_meta.append({"cell": i, "type": "raw", "skip": True})
            continue

        if cell_type == "code":
            content_parts.append(f"<!-- cell:{i} type:code -->")
            content_parts.append(f"```python\n{source}\n```")

            # Include text outputs, omit images
            for output in cell.get("outputs", []):
                output_type = output.get("output_type", "")

                if output_type == "stream":
                    text = "".join(output.get("text", []))
                    if text.strip():
                        content_parts.append(f"Output:\n```\n{text.rstrip()}\n```")

                elif output_type in ("display_data", "execute_result"):
                    data = output.get("data", {})
                    if "text/plain" in data:
                        text = "".join(data["text/plain"])
                        # Skip figure repr strings
                        if text.strip() and not text.strip().startswith("<"):
                            content_parts.append(
                                f"Output:\n```\n{text.rstrip()}\n```"
                            )
                    if "image/png" in data or "image/jpeg" in data:
                        # Get caption from metadata if available
                        caption = ""
                        md = data.get("text/markdown", "")
                        if md:
                            caption = "".join(md) if isinstance(md, list) else md
                        content_parts.append(
                            f"[Figure{': ' + caption.strip() if caption else ''}]"
                        )

            cells_meta.append({
                "cell": i,
                "type": "code",
                "html_id": f"cell-{i}",
            })
            content_parts.append("")  # blank line

        elif cell_type == "markdown":
            content_parts.append(f"<!-- cell:{i} type:markdown -->")
            content_parts.append(source)

            # Extract headings
            headings = []
            for line in source.split("\n"):
                m = re.match(r'^(#{1,6})\s+(.+)', line)
                if m:
                    level = len(m.group(1))
                    text = m.group(2).strip()
                    headings.append({"slug": slugify(text), "level": level})

            cells_meta.append({
                "cell": i,
                "type": "markdown",
                "headings": headings,
            })
            content_parts.append("")  # blank line

    content_md = "\n".join(content_parts)

    cells_json = {
        "version": 1,
        "source_type": "ipynb",
        "title": title,
        "cells": cells_meta,
    }

    return content_md, cells_json


def process_qmd_or_md(file_path):
    """Generate _content.md and cells.json for a .qmd or .md post.

    Parses the file into a sequence of cells (markdown sections and code blocks),
    using the same <!-- cell:N type:code|markdown --> marker format as notebooks.
    Code blocks are fenced (```lang) or executable ({lang}).
    Markdown sections break at h2/h3 headings.
    """
    with open(file_path) as f:
        text = f.read()

    # Strip YAML frontmatter
    title = ""
    body = text
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            frontmatter = parts[1]
            body = parts[2].strip()
            m = re.search(
                r'^title:\s*["\']?(.+?)["\']?\s*$', frontmatter, re.MULTILINE
            )
            if m:
                title = m.group(1)

    # Parse body into cells: markdown sections and code blocks
    # A new cell starts at: a heading (h2/h3), or a fenced/executable code block
    cells = []  # list of (type, text, lang)
    current_lines = []
    in_code_block = False
    code_lang = ""

    def flush_markdown():
        """Save accumulated markdown lines as a cell."""
        nonlocal current_lines
        content = "\n".join(current_lines).strip()
        if content:
            cells.append(("markdown", content, ""))
        current_lines = []

    for line in body.split("\n"):
        if in_code_block:
            current_lines.append(line)
            # Closing fence: ``` with only optional whitespace
            if re.match(r'^```\s*$', line):
                # Save the code block (strip the fences for _content.md)
                code_text = "\n".join(current_lines[1:-1])  # exclude opening/closing ```
                cells.append(("code", code_text, code_lang))
                current_lines = []
                in_code_block = False
        else:
            # Check for opening code fence: ```lang or ```{lang}
            fence = re.match(r'^```\{?(\w+)\}?\s*$', line)
            if fence:
                flush_markdown()
                current_lines = [line]
                code_lang = fence.group(1)
                in_code_block = True
            # Check for bare opening fence ``` (no language)
            elif re.match(r'^```\s*$', line) and not in_code_block:
                flush_markdown()
                current_lines = [line]
                code_lang = ""
                in_code_block = True
            # Check for h2/h3 heading — start new markdown section
            elif re.match(r'^#{2,3}\s+', line):
                flush_markdown()
                current_lines = [line]
            else:
                current_lines.append(line)

    # Flush remaining content
    if in_code_block:
        # Unclosed code block — treat as markdown
        cells.append(("markdown", "\n".join(current_lines).strip(), ""))
    else:
        flush_markdown()

    # Build _content.md and cells.json from parsed cells
    content_parts = []
    cells_meta = []

    for i, (cell_type, cell_text, lang) in enumerate(cells):
        if cell_type == "code":
            content_parts.append(f"<!-- cell:{i} type:code -->")
            lang_tag = lang if lang else ""
            content_parts.append(f"```{lang_tag}\n{cell_text}\n```")
            content_parts.append("")

            cells_meta.append({
                "cell": i,
                "type": "code",
                "lang": lang,
            })
        else:
            content_parts.append(f"<!-- cell:{i} type:markdown -->")
            content_parts.append(cell_text)
            content_parts.append("")

            headings = []
            for hline in cell_text.split("\n"):
                m = re.match(r'^(#{1,6})\s+(.+)', hline)
                if m:
                    level = len(m.group(1))
                    heading_text = m.group(2).strip()
                    headings.append({"slug": slugify(heading_text), "level": level})

            cells_meta.append({
                "cell": i,
                "type": "markdown",
                "headings": headings,
            })

    source_type = "qmd" if str(file_path).endswith(".qmd") else "md"

    cells_json = {
        "version": 1,
        "source_type": source_type,
        "title": title,
        "cells": cells_meta,
    }

    return "\n".join(content_parts), cells_json


def main():
    parser = argparse.ArgumentParser(description="Generate LLM context files")
    parser.add_argument(
        "--site-dir",
        default="_site",
        help="Path to rendered site directory (default: _site)",
    )
    parser.add_argument(
        "--posts-dir",
        default="posts",
        help="Path to posts source directory (default: posts)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    posts_dir = project_root / args.posts_dir
    site_dir = project_root / args.site_dir

    if not posts_dir.exists():
        print(f"Error: posts directory not found: {posts_dir}")
        sys.exit(1)

    if not site_dir.exists():
        print(f"Warning: site directory not found: {site_dir}")
        print("Run 'quarto render' first, then re-run this script.")
        sys.exit(1)

    llms_entries = []
    processed = 0
    errors = 0

    # Find all posts
    for item in sorted(posts_dir.iterdir()):
        if item.name.startswith(".") or item.name.startswith("_"):
            continue

        source_file = None
        post_slug = None

        if item.is_dir():
            # Check for index.ipynb, index.qmd, or index.md
            for candidate in ["index.ipynb", "index.qmd", "index.md"]:
                path = item / candidate
                if path.exists():
                    source_file = path
                    post_slug = item.name
                    break
        elif item.suffix in (".md", ".qmd"):
            # Skip listing pages and special files
            if item.name in ("index.qmd", "index.md"):
                continue
            source_file = item
            post_slug = item.stem

        if source_file is None:
            continue

        print(f"Processing {post_slug}...", end=" ")

        try:
            if source_file.suffix == ".ipynb":
                content_md, cells_json = process_notebook(source_file)
            else:
                content_md, cells_json = process_qmd_or_md(source_file)
        except Exception as e:
            print(f"ERROR: {e}")
            errors += 1
            continue

        # Write to _site/posts/<slug>/ directory
        out_dir = site_dir / "posts" / post_slug
        out_dir.mkdir(parents=True, exist_ok=True)

        content_path = out_dir / "_content.md"
        cells_path = out_dir / "cells.json"

        with open(content_path, "w") as f:
            f.write(content_md)
        with open(cells_path, "w") as f:
            json.dump(cells_json, f, indent=2)

        print(f"OK ({cells_json['source_type']}, {len(cells_json['cells'])} cells)")
        processed += 1

        # Add to llms.txt
        url = f"{SITE_URL}/posts/{post_slug}/_content.md"
        entry_title = cells_json.get("title", post_slug) or post_slug
        llms_entries.append(f"{entry_title}: {url}")

    # Write llms.txt
    llms_path = site_dir / "llms.txt"
    with open(llms_path, "w") as f:
        f.write(f"# {SITE_URL}\n")
        f.write("# LLM-readable content index\n")
        f.write(f"# Generated by generate_llm_context.py\n\n")
        for entry in sorted(llms_entries):
            f.write(entry + "\n")

    print(f"\nDone: {processed} posts processed, {errors} errors")
    print(f"Generated {llms_path.relative_to(project_root)} with {len(llms_entries)} entries")


if __name__ == "__main__":
    main()
