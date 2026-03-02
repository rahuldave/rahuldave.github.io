# Skill: Finalize a New Blog Post

Run post-creation checks and generate a card image for any new or updated blog post.

## Invocation

```
/finalize-post <post_path>
```

- `post_path`: path to the post file relative to project root (e.g. `posts/seasons/index.qmd`, `posts/distributions.md`, `posts/probability/index.ipynb`)

## Procedure

### 1. Validate Frontmatter

Read the post file and verify all required frontmatter fields are present:

- `title` — post title
- `subtitle` — catchy one-liner for listing cards
- `description` — two-sentence content summary
- `categories` — lowercase tags from `_categories.txt`
- `date` — YYYY-MM-DD format

If any field is missing, generate a suggestion and present it to the user for approval using `AskUserQuestion`.

### 2. Validate Categories Against `_categories.txt`

1. **Read `_categories.txt`** from the project root (canonical list, one per line, sorted).
2. **Check each category** in the post's frontmatter against the canonical list.
3. **If any category is not in the list**, present it to the user: propose mapping to an existing category, or adding it as new.
4. **After approval, update `_categories.txt`** with any new categories (keep sorted, one per line).

### 3. Render and Verify

```bash
quarto render <post_path>
```

Check that the render succeeds without errors.

### 4. Card Image for Posts Without Content Images

Check whether the rendered post has an auto-discovered thumbnail image (Quarto extracts these from notebook output cells and inline images). If it does, skip this step.

**How to check**: look for an `image:` field already in frontmatter, or for embedded images in the post content (notebook output cells with `image/png`, markdown `![](...)` images, etc.).

If the post has **no content images** (e.g. interactive Three.js/JS demos, pure-text posts):

1. **Render the post** if not already done: `quarto render <post_path>`
2. **Use `agent-browser`** to open the rendered HTML file from `_site/`.
3. **Wait for full render**, especially for JS-heavy posts. Use `agent-browser wait 2000` or similar.
4. **Hide overlay UI** if present: `agent-browser eval "document.querySelectorAll('[style*=\"z-index\"]').forEach(el => el.style.display = 'none')"`
5. **Take a screenshot**: `agent-browser screenshot /tmp/card-screenshot.png`
6. **Crop** to the key visual using `sips` (macOS):
   ```bash
   sips -c <height> <width> /tmp/card-screenshot.png --out <post_folder>/assets/card.png
   ```
   Center the crop on the main visual element. Target roughly 600x400 or similar card-friendly aspect ratio.
7. **Add `image: assets/card.png`** to the post's YAML frontmatter.
8. **Close the browser**: `agent-browser close`

### 5. Bundle Notebook Post

If the post is a notebook (`index.ipynb`), invoke the `/bundle-post` skill to:
- Check for missing data files and copy them in
- Inject PEP 723 dependency metadata for `juv`
- Verify all referenced files are present

Skip this step for markdown (`.md`) and QMD (`.qmd`) posts.

### 6. Verify Card on Listing Page

1. **Re-render** the index page: `quarto render index.qmd`
2. **Use `agent-browser`** to open `_site/index.html` (or `_site/posts.html`).
3. **Take a screenshot** and verify the post's card shows the thumbnail correctly in the grid.
4. **Close the browser**.

### 7. Update CLAUDE.md (if importing from AM207)

If this post was imported from the AM207 wiki, update the "Content Import Status" section in `CLAUDE.md` to reflect the new import. Otherwise skip this step.

## Notes

- This skill is meant to be invoked after creating any new post, whether from wiki import or from scratch.
- The `/import-wiki-notes` skill calls this skill at the end for each imported post.
- For Three.js or JS-heavy interactive content, use `.qmd` format — see `posts/seasons/index.qmd` for the pattern.
- Three.js / interactive containers should have `padding: 1.5rem` to avoid the canvas being flush against the container edges. Use `min-height: 580px` or more for comfortable viewing.
- All categories must be lowercase.
