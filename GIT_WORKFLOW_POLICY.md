# ===========================================================================
# GIT WORKFLOW POLICY — Cerberus / LamiaFabrica
# ===========================================================================
# Agent enforcement: All automated changes must follow this workflow.
# Human override: Allowed for emergencies, but must be documented.
# ===========================================================================

## 1. BEFORE ANY WORK

- [ ] `git status` — confirm clean working tree or intended changes only
- [ ] `git log --oneline -3` — verify we're on the right branch (main)
- [ ] If working tree is dirty: commit or stash before proceeding

## 2. DURING WORK

- [ ] Make atomic commits: one logical change per commit
- [ ] Commit message format: `<area>: <what> (<why if non-obvious>)`
  - Good: `npu_encoder: add HailoRT HEF loading path`
  - Bad:  `updates`
- [ ] NEVER commit:
  - Build outputs (`build/`, `*.exe`, `*.o`, `*.obj`)
  - IDE files (`.vscode/`, `.vs/`, `*.user`)
  - OS junk (`Thumbs.db`, `.DS_Store`, `*Zone.Identifier`)
  - Debug symbols >50 MB (`*.pdb` in `ort/lib/`, vendor SDKs)
  - Temporary or scratch directories
- [ ] ALWAYS update `.gitignore` when adding new generated artifact types
- [ ] ALWAYS update `.gitattributes` for new binary file types (LFS policy)

## 3. LFS SIZE GUARDRAILS (GitHub hard limits)

| Limit | Value | Policy |
|-------|-------|--------|
| Single file (git) | 100 MB | NEVER commit raw files >90 MB |
| Single file (LFS free) | 2 GB | Prefer <500 MB per file |
| LFS storage (free) | 1 GB total | Monitor with `git lfs ls-files` |
| Repo size (hard) | ~5 GB | Aggressive pruning of old artifacts |

**Prohibited from git entirely:**
- ORT debug symbols (`ort/lib/*.pdb`) — 200-350 MB each
- Compiler toolchains (`gcc-*/`) — use download scripts instead
- Model weights >1 GB — use huggingface.co or S3, not git

## 4. BEFORE PUSH

- [ ] `git diff --stat` — review what files are being pushed
- [ ] `git lfs status` — confirm no unexpected LFS objects
- [ ] `git log --oneline origin/main..HEAD` — review commit count and messages
- [ ] Run build: `py build.py --no-cmake` — must pass 0 errors
- [ ] If build fails: fix before pushing (never push broken code)

## 5. PUSH

```bash
# Normal push (fast-forward)
git push origin main

# If rejected (diverged history)
#   1. PULL first: git pull --rebase origin main
#   2. Resolve conflicts
#   3. Re-test build
#   4. Push again

# Force push ONLY when:
#   - History rewrite (filter-branch/filter-repo) to purge large files
#   - Rebasing a feature branch that has never been shared
#   NEVER force-push to shared branches without team agreement
```

## 6. AFTER PUSH

- [ ] Verify CI/build passes on GitHub
- [ ] If LFS quota warning appears: audit with `git lfs ls-files`, purge orphans
- [ ] Document breaking changes in `research/` or `CHANGELOG.md`

## 7. EMERGENCY RECOVERY

**PDB / large file accidentally pushed (blocks all future pushes):**
```bash
# 1. Backup repo
cp -r repo repo-backup-$(date +%Y%m%d)

# 2. Purge from ALL history
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch <path>' \
  --prune-empty --tag-name-filter cat -- --all

# 3. Clean up
git update-ref -d refs/original/refs/heads/main
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# 4. Force-push
git push --force origin main
```

## 8. AGENT CHECKLIST (Auto-enforced)

Every coding session ends with:
1. `git add` only intended files
2. `git status` confirms no unexpected additions
3. `git commit` with descriptive message
4. `py build.py --no-cmake` passes
5. `git push origin main` succeeds without errors
6. Update `.gitignore` or `.gitattributes` if new patterns discovered

---
Enforcement: This policy is checked at session end.
Violations: Logged to `research/git_policy_violations.md` for review.
