# Skill: Caption Non-Generated Images

Add captions (with optional citations) to manually embedded images in blog posts.

## Invocation

```
/caption-images [post_path]
```

- `post_path` (optional): path to a specific post (e.g. `posts/frequentist.md`, `posts/MLE/index.ipynb`). If omitted, scans all posts.

## When to Use

- After importing a new post via `/import-wiki-notes`
- When reviewing existing posts for missing captions
- The `/import-wiki-notes` skill should call this at the end for each imported post

## Procedure

### 1. Scan for Non-Generated Images

For each post file (`.md`, `.qmd`, or `.ipynb`):

**Markdown/QMD files**: Find all `![...](path)` references.

**Notebooks**: Find all `![...](path)` references in **markdown cells only** (cell_type: "markdown").

**Exclude** these (they are generated, not manually embedded):
- Images in `*_files/` directories (notebook output cells)
- `data:` URIs
- Images that are only referenced in `image:` frontmatter field (card thumbnails)

**Also exclude** numbered slide images (e.g. `1.png`, `2.png`, `11.png` — files in `assets/` directories whose name is just a number). These are presentation slides, not figures that need captions.

### 2. Check Which Images Already Have Captions

An image **already has a caption** if the text between `![` and `]` is non-empty and descriptive (not just a filename or code reference like `m:bishopprob`).

Collect only images that need captions (empty alt text or non-descriptive alt text).

### 3. View Each Image

Read each image file to understand what it depicts. This is essential for generating a good suggested caption.

### 4. Render Preview and Ask User for Captions (per post)

**Before asking about each post's images**, render a preview so the user can see the post in context:
```bash
quarto render <post_path>
open _site/<post_url>
```
For example: `quarto render posts/MLE/index.ipynb && open _site/posts/MLE/index.html`

This lets the user see where each image appears and what context surrounds it, which is essential for writing good captions with citations.

Then use `AskUserQuestion` with up to **4 questions per call** (one per image). Only ask about images from the **currently previewed post** — do not mix images from different posts in the same batch.

**IMPORTANT: Always ask the user for captions. Do NOT auto-generate captions without user approval.** Even if you can describe the image, the user may want specific wording, citations, or to skip captioning entirely.

Each question should:
- **Question text**: Show the post filename, image position (1st, 2nd, etc.), and image filename
- **Header**: Short label (image filename, max 12 chars)
- **Option 1**: Your suggested descriptive caption (based on viewing the image and reading surrounding context). This is the pre-filled suggestion.
- **Option 2**: "No caption needed" — skip this image

The user will either:
- **Select Option 1** to accept the caption as-is
- **Select "Other"** to type a modified caption (typically adding a `[Source]` citation)
- **Select Option 2** to skip

**Format for captions with citations**: `Descriptive caption text [Source: Author or Book]`

### 5. Apply Captions

**For markdown/QMD files**: Replace `![](path)` or `![old text](path)` with `![New caption](path)`.

**For notebooks**: Use `_scripts/update_captions.py` to update captions:
```bash
python3 _scripts/update_captions.py posts/MLE/index.ipynb \
    --replace '![](assets/gaussmle.png)' '![My caption](assets/gaussmle.png)' \
    --replace '![](assets/linregmle.png)' '![Another caption](assets/linregmle.png)'
```
The script only modifies markdown cells and can also accept replacements via a JSON file (`--json`).

**WARNING**: The `--replace` args must be the full markdown image syntax (e.g. `![](path)` → `![caption](path)`). Do NOT pass just the path and caption as separate args — that will replace the path string with the caption text, corrupting the image reference. Alternatively, use inline Python to do the replacements directly on the notebook JSON to avoid shell quoting issues with long captions.

### 6. Verify

After all captions are applied, do a quick scan to confirm no images were missed:
```bash
# For markdown files
grep -n '!\[\](' posts/<name>.md

# For notebooks
python3 -c "
import json, glob
for nb in glob.glob('posts/*/index.ipynb'):
    data = json.load(open(nb))
    for i, c in enumerate(data['cells']):
        if c['cell_type'] == 'markdown':
            src = ''.join(c['source'])
            if '![](' in src:
                print(f'{nb} cell {i}: empty caption found')
"
```

## Example

For `posts/frequentist.md` with image `gaussmle.png` showing two Gaussian curves:

Question: `frequentist.md — 2nd image: gaussmle.png. What caption?`
- Option 1: `Two Gaussians illustrating maximum likelihood estimation`
- Option 2: `No caption needed`

User selects Other and types: `Two Gaussians illustrating maximum likelihood estimation [Source: AM207]`

Result in file: `![Two Gaussians illustrating maximum likelihood estimation [Source: AM207]](images/gaussmle.png)`

## Notes

- Captions in Quarto become visible figure captions below the image when the image is alone in a paragraph
- `{fig-alt="..."}` can be added after the image for separate accessibility alt text if desired
- This skill is non-destructive — it only modifies the text between `![` and `]`
- Numbered slide images (e.g. `assets/vizasstory/1.png`) are excluded automatically
