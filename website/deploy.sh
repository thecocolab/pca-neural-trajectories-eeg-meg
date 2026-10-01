#!/usr/bin/env bash
# Rebuild the site and publish it to the gh-pages branch.
#
#   ./website/deploy.sh
#
# The branch is rewritten as a single orphan commit each time, so the report HTML
# (a few hundred MB) never accumulates in history. Nothing else on gh-pages is preserved.

set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
worktree="$(mktemp -d)"
build_branch="gh-pages-build-$$"   # throwaway; works whether or not a local gh-pages exists
trap 'git -C "$repo" worktree remove --force "$worktree" 2>/dev/null || true; rm -rf "$worktree"; git -C "$repo" branch -qD "$build_branch" 2>/dev/null || true' EXIT

python3 "$repo/website/build.py" "$@"

git -C "$repo" worktree add --detach "$worktree" >/dev/null
git -C "$worktree" checkout --orphan "$build_branch" >/dev/null 2>&1
git -C "$worktree" rm -rqf . >/dev/null 2>&1 || true

cp -R "$repo/website/index.html" "$repo/website/view.html" "$repo/website/reports" "$worktree/"
touch "$worktree/.nojekyll"   # serve paths verbatim; skip the Jekyll build

git -C "$worktree" add -A
git -C "$worktree" commit -qm "Publish analysis reports ($(date +%Y-%m-%d))"
git -C "$worktree" push -qf origin HEAD:gh-pages

echo "Published $(find "$repo/website/reports" -name '*.html' | wc -l | tr -d ' ') reports to gh-pages."
