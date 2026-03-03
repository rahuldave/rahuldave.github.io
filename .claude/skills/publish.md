# Skill: Publish Site

Build the site, commit all changes, and push to GitHub (which triggers GitHub Pages deployment).

## Invocation

```
/publish [commit message]
```

- `commit message` (optional): If provided, use as the commit message. Otherwise, generate one from the staged changes.

## Procedure

### 1. Build

Run `make build` which renders the site to `_site/`, generates LLM context files (`_content.md`, `cells.json`, `llms.txt`), and generates zip bundles (`<slug>.zip`, `bundles.json`). Uses stamp files so unchanged stages are skipped.

### 2. Review Changes

Run `git status` and `git diff --stat` to see what changed. Summarize the changes for the user.

### 3. Commit

- Stage all changed and new files (source posts, images, data, scripts, CLAUDE.md, etc.)
- **Do NOT stage** files in `.gitignore` (e.g. `_site/`, `.claude/`)
- Generate a concise commit message from the changes if none was provided
- Append `Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>`
- If there are no changes to commit, inform the user and stop

### 4. Push

Run `make deploy` to build and push `_site/` to the `gh-pages` branch. Or push `main` separately if only committing source changes.

### 5. Confirm

Show the user the commit hash and confirm the push succeeded.
