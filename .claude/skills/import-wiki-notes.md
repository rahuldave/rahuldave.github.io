# Skill: Import Wiki Notes as Blog Posts

Import educational content from the AM207 wiki (or similar source) into the Quarto blog.

## Invocation

```
/import-wiki-notes <source_folder> <note_name1> [note_name2] ...
```

- `source_folder`: path to the wiki folder containing `.md` and `.ipynb` files (e.g. `/Users/rahul/Attic/Projects/AM207/2018fall_wiki/wiki`)
- `note_name1`, etc.: base names without extension (e.g. `probability`, `distributions`, `distrib-example`)

## Procedure

For each note name, follow these steps:

### 1. Determine Source Type

Check the source folder for both `<name>.md` and `<name>.ipynb`.

- **If `.ipynb` exists**: the notebook is the primary source. The `.md` is just a rendered view — ignore it.
- **If only `.md` exists**: the markdown file is the source.

### 2. Find Embedded Images and Data

Scan the source file for:
- **Markdown images**: `![...](path)` where path is NOT a `data:` URI and NOT a `*_files/*` path (those are notebook output cells, not educational images)
- **Data files**: `read_csv('path')`, `open('path')`, `load('path')`, `loadtxt('path')`, `genfromtxt('path')` patterns in code cells. Check **both** `assets/` and `data/` subdirectory references.

Collect all referenced file paths. These are typically relative like `images/foo.png` or `data/bar.csv`.

For data files, also check the wiki `data/` folder (`~/Attic/Projects/AM207/2018fall_wiki/wiki/data/`) for any files referenced by the notebook. Copy them into the post's `data/` subdirectory (create it if needed).

### 3. Copy Files Into Posts

**For notebooks** (`.ipynb`): Use `_scripts/import_notebook.py` to automate this:
```bash
python3 _scripts/import_notebook.py \
    --source <wiki_path>/<name>.ipynb \
    --name <name> \
    --title "Title" --subtitle "Subtitle" --description "Description" \
    --categories cat1 cat2 \
    --date YYYY-MM-DD \
    --images img1.png img2.png
```
This creates `posts/<name>/index.ipynb` with a raw frontmatter cell prepended, fixes `images/` → `assets/` paths, copies listed images, and sets notebook metadata. Run `python3 _scripts/import_notebook.py --help` for full usage.

**For markdown** (`.md` only): Copy manually and edit frontmatter:
```
posts/<name>.md        # copied from source (flat file)
posts/images/          # shared images go here
  foo.png
posts/data/            # shared data goes here (if needed)
  bar.csv
```

### 4. Update Paths Inside the Files

**Notebooks**: `_scripts/import_notebook.py` handles this automatically. If doing manually, rewrite cell sources:
- `images/X.png` -> `assets/X.png`
- `./images/X.png` -> `assets/X.png`
- `data/X.csv` -> `assets/X.csv`
- `'data/X.csv'` -> `'assets/X.csv'`
- `"data/X.csv"` -> `"assets/X.csv"`

**Markdown files**: Update image paths:
- `images/X.png` stays as `images/X.png` (resolves to `posts/images/X.png`)

### 5. Fix Internal Links

Replace any `*.html` links that point to other wiki notes with appropriate Quarto paths:
- `(probability.html)` -> `(probability/index.ipynb)` (if target is a notebook post)
- `(distributions.html)` -> `(distributions.md)` (if target is a markdown post)
- `(SamplingCLT.html)` -> leave as-is or remove if the target doesn't exist yet

### 6. Remove Wiki/Jekyll Artifacts

After importing, clean up wiki artifacts from the notebook cells. This must happen **after** the import script runs (which prepends the raw frontmatter cell), so cell indices are relative to the imported notebook.

**Remove these from the first markdown cell (cell 1, after the raw frontmatter cell):**
- **Duplicate title heading**: A `# Title` line that duplicates the frontmatter title — remove it
- **Keywords line**: `##### Keywords: ...` lines — remove them (keywords are absorbed into frontmatter categories)
- If the cell becomes empty after removal, delete the entire cell

**Remove these as separate cells (typically cell 2 or 3):**
- **TOC cells**: cells containing `{:.no_toc}` and `{: toc}` — delete the entire cell

**Remove from any cell:**
- Jekyll template tags: `{% assign ... %}`, `{% include ... %}` — remove these lines
- `layout: wiki` or `layout: default` in frontmatter — remove

### 7. Add/Fix Quarto Frontmatter

**Notebooks**: The first cell must be a `raw` cell (cell_type: "raw") with YAML:
```yaml
---
title: "<Title>"
subtitle: "<Catchy one-liner for listing cards>"
description: "<Two-sentence content summary>"
categories:
    - <category1>
    - <category2>
date: <YYYY-MM-DD>
format:
    html: default
---
```

The `format` block is for Quarto rendering. **Do not include `ipynb: default`** — downloadable notebooks are handled by zip bundles instead (see `/bundle-post` skill). `_scripts/import_notebook.py` generates this frontmatter automatically.

If the notebook already has a markdown cell with a `# Title` and `##### Keywords:` line, use those to derive title and categories, then replace that cell with the raw frontmatter cell.

**Markdown files**: Replace the existing YAML frontmatter with Quarto-compatible YAML (same fields as above, no `layout`, no `shorttitle`, no `keywords` list).

For both: generate the `subtitle` (catchy one-liner) and `description` (2-sentence summary) by reading and summarizing the full content of the post.

### 7b. Present Subtitle and Description for Approval

Before writing the frontmatter, **present the proposed subtitle and description to the user for approval** using `AskUserQuestion` or inline options. For each notebook, show:
- **2 subtitle options** (one-sentence ledes for listing cards)
- **2 description options** (two-sentence summaries for the post page)

Wait for the user to pick or provide alternatives before finalizing the frontmatter.

**IMPORTANT: All categories must be lowercase.** Convert any mixed-case keywords/categories to lowercase (e.g. `Statistics` -> `statistics`, `MonteCarlo` -> `montecarlo`, `Visualization` -> `visualization`). This prevents duplicate tags in Quarto listings.

### 7c. Remove Keywords Line

**Remove the `##### Keywords:` line** (and any blank line immediately after it) from the notebook's markdown cell. The keywords have been absorbed into the frontmatter categories — they should not remain as visible content in the post.

### 8. Check Referenced Images Exist

```bash
python3 -c "
import json, os, re, glob
for nb in glob.glob('posts/*/index.ipynb'):
    data = json.load(open(nb))
    folder = os.path.dirname(nb)
    for c in data['cells']:
        src = ''.join(c['source'])
        for m in re.finditer(r'!\[.*?\]\((?!data:)(.*?)\)', src):
            img = os.path.join(folder, m.group(1))
            if not os.path.exists(img):
                print(f'MISSING: {img} (from {nb})')
"
```

### 9. Caption Non-Generated Images

Invoke the `/caption-images` skill for each imported post. This presents each manually embedded image to the user and asks for a descriptive caption with optional citation.

### 10. Finalize Post

Invoke the `/finalize-post` skill for each imported post. This handles:
- Category validation against `_categories.txt`
- Rendering and verification
- Card image generation for posts without content images
- Listing page verification
- **Bundle generation** for notebook posts (PEP 723 deps + zip bundle)

## Notes

- Never copy notebook output images (`*_files/*.png`) — those are regenerated by Quarto
- The `assets/` folder is per-notebook; `posts/images/` and `posts/data/` are shared across flat `.md` posts
- For Three.js or other JS-heavy interactive content, use `.qmd` format instead of `.ipynb` — see `posts/seasons/index.qmd` for the pattern (load JS via `include-in-header`, use `::: {}` fenced divs, inline `<script>`)
- `.claude/` is in `.gitignore` so this skill is not checked in
