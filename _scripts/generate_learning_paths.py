#!/usr/bin/env python3
"""Generate learning path pages and JSON manifest from YAML DAGs.

Reads _learning-paths/*.yml, topologically sorts nodes within parts, and produces:
  - learning-paths/<id>.qmd  (per-path listing page with part headers)
  - assets/learning-paths.json  (manifest for runtime JS, including part info)
"""

import heapq
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LP_DIR = ROOT / "_learning-paths"
OUT_DIR = ROOT / "learning-paths"
POSTS_DIR = ROOT / "posts"
MANIFEST_PATH = ROOT / "assets" / "learning-paths.json"

LEVEL_ORDER = {"beginner": 1, "intermediate": 2, "advanced": 3}


# ── Minimal YAML parser (no PyYAML dependency) ────────────────────────


def parse_yaml(text: str) -> dict:
    """Parse our fixed YAML format: top-level scalars + parts list with nodes."""
    result: dict = {}
    parts: list[dict] = []
    current_part: dict | None = None
    current_node: dict | None = None
    in_parts = False
    in_nodes = False

    for raw_line in text.splitlines():
        line = raw_line.rstrip()

        # Skip empty lines and comments
        if not line or line.lstrip().startswith("#"):
            continue

        # Detect parts: section
        if re.match(r"^parts:\s*$", line):
            in_parts = True
            in_nodes = False
            continue

        if not in_parts:
            # Top-level scalar: key: value
            m = re.match(r"^(\w+):\s*(.+)$", line)
            if m:
                key, val = m.group(1), m.group(2).strip().strip('"').strip("'")
                result[key] = val
            continue

        # Inside parts list
        # New part: "  - title: ..."
        m = re.match(r"^\s+-\s+title:\s*(.+)$", line)
        if m:
            current_part = {
                "title": m.group(1).strip().strip('"').strip("'"),
                "description": "",
                "nodes": [],
            }
            parts.append(current_part)
            current_node = None
            in_nodes = False
            continue

        # Part description: "    description: ..."
        m = re.match(r"^\s+description:\s*(.+)$", line)
        if m and current_part is not None and not in_nodes:
            current_part["description"] = m.group(1).strip().strip('"').strip("'")
            continue

        # Nodes sub-list: "    nodes:"
        if re.match(r"^\s+nodes:\s*$", line) and current_part is not None:
            in_nodes = True
            continue

        if in_nodes and current_part is not None:
            # New node: "      - slug: foo"
            m = re.match(r"^\s+-\s+slug:\s*(\S+)", line)
            if m:
                current_node = {"slug": m.group(1), "requires": [], "order": None}
                current_part["nodes"].append(current_node)
                continue

            # Node property: "        requires: [a, b]"
            m = re.match(r"^\s+requires:\s*\[([^\]]*)\]", line)
            if m and current_node is not None:
                deps = [s.strip() for s in m.group(1).split(",") if s.strip()]
                current_node["requires"] = deps
                continue

            # Node property: "        order: 3"
            m = re.match(r"^\s+order:\s*(\d+)", line)
            if m and current_node is not None:
                current_node["order"] = int(m.group(1))
                continue

    result["parts"] = parts
    return result


# ── Frontmatter extraction ─────────────────────────────────────────────


def _extract_frontmatter_field(slug: str, field: str) -> str | None:
    """Extract a named field from a post's frontmatter."""
    # Try notebook first
    nb_path = POSTS_DIR / slug / "index.ipynb"
    if nb_path.exists():
        with open(nb_path) as f:
            nb = json.load(f)
        if nb.get("cells"):
            cell = nb["cells"][0]
            if cell.get("cell_type") == "raw":
                source = "".join(cell.get("source", []))
                for line in source.splitlines():
                    line = line.strip()
                    if line.startswith(f"{field}:"):
                        return line.split(":", 1)[1].strip().strip('"').strip("'")
        return None

    # Try markdown
    md_path = POSTS_DIR / f"{slug}.md"
    if md_path.exists():
        in_frontmatter = False
        with open(md_path) as f:
            for line in f:
                line = line.rstrip()
                if line == "---":
                    if not in_frontmatter:
                        in_frontmatter = True
                        continue
                    else:
                        break
                if in_frontmatter and line.strip().startswith(f"{field}:"):
                    return line.split(":", 1)[1].strip().strip('"').strip("'")
        return None

    return None


def get_post_title(slug: str) -> str:
    return _extract_frontmatter_field(slug, "title") or slug


def get_post_description(slug: str) -> str:
    return _extract_frontmatter_field(slug, "description") or ""


def get_post_url(slug: str) -> str:
    """Return the URL for a post slug."""
    if (POSTS_DIR / slug / "index.ipynb").exists():
        return f"/posts/{slug}/"
    if (POSTS_DIR / slug / "index.qmd").exists():
        return f"/posts/{slug}/"
    # Markdown flat file
    return f"/posts/{slug}"


# ── Topological sort (Kahn's with tie-breaking) ────────────────────────


def toposort(nodes: list[dict]) -> list[dict]:
    """Kahn's algorithm with tie-breaking: order field (lower first), then slug."""
    slug_to_node = {n["slug"]: n for n in nodes}
    in_degree = {n["slug"]: 0 for n in nodes}
    dependents: dict[str, list[str]] = {n["slug"]: [] for n in nodes}

    for node in nodes:
        for req in node["requires"]:
            if req not in slug_to_node:
                print(f"ERROR: '{req}' in requires of '{node['slug']}' is not a node in this path", file=sys.stderr)
                sys.exit(1)
            in_degree[node["slug"]] += 1
            dependents[req].append(node["slug"])

    # Priority: (order or 999, slug)
    heap = []
    for slug, deg in in_degree.items():
        if deg == 0:
            n = slug_to_node[slug]
            heapq.heappush(heap, (n["order"] or 999, slug))

    result = []
    while heap:
        _, slug = heapq.heappop(heap)
        result.append(slug_to_node[slug])
        for dep in dependents[slug]:
            in_degree[dep] -= 1
            if in_degree[dep] == 0:
                n = slug_to_node[dep]
                heapq.heappush(heap, (n["order"] or 999, dep))

    if len(result) != len(nodes):
        sorted_slugs = {n["slug"] for n in result}
        missing = [n["slug"] for n in nodes if n["slug"] not in sorted_slugs]
        print(f"ERROR: Cycle detected involving: {missing}", file=sys.stderr)
        sys.exit(1)

    return result


# ── Generation ──────────────────────────────────────────────────────────


def generate_path(yml_path: Path, manifest: dict) -> None:
    """Process one YAML path definition."""
    path_id = yml_path.stem  # e.g. "intro-to-sampling"
    config = parse_yaml(yml_path.read_text())

    title = config.get("title", path_id)
    description = config.get("description", "")
    level = config.get("level", "beginner")
    parts = config.get("parts", [])

    if not parts:
        print(f"WARNING: {yml_path.name} has no parts, skipping", file=sys.stderr)
        return

    # Collect all nodes across parts for validation and cross-part requires
    all_nodes = []
    for part in parts:
        all_nodes.extend(part["nodes"])

    # Validate slugs exist as posts
    for node in all_nodes:
        slug = node["slug"]
        if not (POSTS_DIR / slug).is_dir() and not (POSTS_DIR / f"{slug}.md").exists():
            print(f"ERROR: Post '{slug}' not found for path '{path_id}'", file=sys.stderr)
            sys.exit(1)

    # Topological sort across ALL nodes (requires can cross part boundaries)
    sorted_all = toposort(all_nodes)
    sorted_slugs = [n["slug"] for n in sorted_all]

    # Build parts with sorted steps — preserve part grouping but sort within each part
    # by the global topological order
    global_order = {slug: i for i, slug in enumerate(sorted_slugs)}
    total_count = len(sorted_slugs)

    manifest_parts = []
    global_step = 0

    # Load optional content markdown
    content_path = LP_DIR / f"{path_id}-content.md"
    extra_content = content_path.read_text().strip() if content_path.exists() else ""

    # Generate .qmd
    order = LEVEL_ORDER.get(level, 9)
    num_parts = len(parts)
    subtitle = f"{level.title()} \u00b7 {num_parts} parts \u00b7 {total_count} posts"

    # Check for optional image
    image_path = LP_DIR / f"{path_id}-card.png"
    if not image_path.exists():
        image_path = LP_DIR / f"{path_id}-card.jpg"
    has_image = image_path.exists()

    lines = [
        "---",
        f'title: "{title}"',
        f'description: "{description}"',
        f'subtitle: "{subtitle}"',
        f"order: {order}",
    ]
    if has_image:
        # Copy image to output dir so it's next to the .qmd
        import shutil
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        dest = OUT_DIR / image_path.name
        shutil.copy2(image_path, dest)
        lines.append(f"image: {image_path.name}")
    lines.append("---")
    lines.append("")

    if extra_content:
        lines.append(extra_content)
        lines.append("")

    # Build all steps first so we can place Start button before parts
    for part_idx, part in enumerate(parts):
        part_sorted = sorted(part["nodes"], key=lambda n: global_order[n["slug"]])
        part_steps = []
        for node in part_sorted:
            global_step += 1
            slug = node["slug"]
            part_steps.append({
                "slug": slug,
                "title": get_post_title(slug),
                "description": get_post_description(slug),
                "url": get_post_url(slug),
                "global_step": global_step,
            })
        manifest_parts.append({
            "title": part["title"],
            "description": part["description"],
            "steps": part_steps,
        })

    # All steps flat
    all_steps_build = []
    for mp in manifest_parts:
        all_steps_build.extend(mp["steps"])

    first_url = f'{all_steps_build[0]["url"]}?path={path_id}&step=1'
    lines.append(f"[Start this path]({first_url}){{.btn .btn-primary}}")
    lines.append("")

    # Emit parts with card-style post entries
    for part_idx, mp in enumerate(manifest_parts):
        part_title = mp["title"]
        part_description = mp["description"]

        lines.append(f"## Part {part_idx + 1}: {part_title}")
        lines.append("")
        if part_description:
            lines.append(part_description)
            lines.append("")

        for step in mp["steps"]:
            link = f'{step["url"]}?path={path_id}&step={step["global_step"]}'
            lines.append(f'::: {{.lp-step-card}}')
            lines.append(f'**{step["global_step"]}.** [{step["title"]}]({link})')
            lines.append("")
            if step["description"]:
                lines.append(f'{step["description"]}')
            lines.append(":::")
            lines.append("")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    qmd_path = OUT_DIR / f"{path_id}.qmd"
    qmd_path.write_text("\n".join(lines))
    print(f"  Generated {qmd_path.relative_to(ROOT)}")

    # Build manifest steps (strip internal global_step)
    all_steps = []
    manifest_parts_clean = []
    for mp in manifest_parts:
        clean_steps = []
        for s in mp["steps"]:
            clean_steps.append({
                "slug": s["slug"],
                "title": s["title"],
                "description": s["description"],
                "url": s["url"],
            })
        all_steps.extend(clean_steps)
        manifest_parts_clean.append({
            "title": mp["title"],
            "description": mp["description"],
            "steps": clean_steps,
        })

    manifest[path_id] = {
        "title": title,
        "description": description,
        "level": level,
        "parts": manifest_parts_clean,
        "steps": all_steps,
    }


def main() -> None:
    yml_files = sorted(LP_DIR.glob("*.yml"))
    if not yml_files:
        print("No learning path YAML files found in _learning-paths/")
        return

    manifest: dict = {}

    print(f"Generating learning paths from {len(yml_files)} YAML file(s)...")
    for yml_path in yml_files:
        generate_path(yml_path, manifest)

    # Write manifest
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"  Wrote {MANIFEST_PATH.relative_to(ROOT)}")
    print(f"Done: {len(manifest)} path(s), {sum(len(p['steps']) for p in manifest.values())} total steps")


if __name__ == "__main__":
    main()
