---
title: "GitButler after a squash-merged PR: the local stack stays conflicted; resolve toward the merged content and insert stacked branches by commit anchor"
author: CodeAndCanvas728
date: 2026-10-30
id: l-gitbutler-squash-merge-cleanup
scope: repo
status: active
src: agent
touches:
  - .trail/learnings/
---

When a PR merges as a **squash** (like #11 → `1bf46f3`), GitButler keeps the local pre-squash history applied on the (unnamed) default workspace branch — 13 commits, most `{conflicted}` because they re-apply their own already-merged content. Symptoms that look like `but` bugs are this state: `but pull` says "Everything is up to date" (it will NOT reconcile — the individual commits don't exist upstream), `but discard <stack-selector>` and ref operations on the unnamed branch fail with "Reference name cannot end with a slash" (no name to compose a ref from), and `but push <new-branch>` refuses while any ancestor commit is conflicted. The working fix loop: `but resolve conflicts <branch>` oldest-first, taking `--ours` (the new base = the merged content) for every file of every stale commit — each becomes a no-op, the loop ends with "no conflicted commits", and the push goes through. The final `but resolve conflicts` call exits non-zero with "has no conflicted commits" — that is success, not failure. To commit new work that depends on a commit inside such a stack, anchor the branch at the stack's tip **commit id** (`but branch new <name> --above <tip-commit-id>`, e.g. `--above pon`) — targeting a commit id inserts mid-stack and keeps sibling branches above (like `fix/app-boot-venv-resilience`, whose commit must NOT ride in the new PR) out of the way; targeting the *branch name* of the unnamed stack is what crashed, and `-b` alone creates an unstacked branch whose ancestry lacks the stack (→ "depends on commit tzm" errors). Net diff check after cleanup: `git diff <merge-base> gitbutler/workspace` showed exactly two leftover junk artifacts (a duplicated `## [Unreleased]` changelog block and duplicated hint constants), which the fix commit then cancelled — so the PR's net diff was only the real fix.
