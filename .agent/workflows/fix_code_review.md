---
description: Fix issues identified by the /code_review workflow
---

This workflow is used **after** running `/code_review` on a file. It systematically addresses and fixes all identified issues with **strict scope control**.

## Prerequisites

- A code review has been performed and documented (either in chat history or in a review artifact)
- The review contains categorized issues (Critical Blockers, Major Concerns, Nitpicks)

## Steps

### 1. Parse the Code Review Output

Identify and categorize all issues from the code review:

1. **🚨 Critical Blockers (CB)** - Must be fixed before any merge
2. **⚠️ Major Concerns (MC)** - Require refactoring
3. **💡 Nitpicks (N)** - Improve code quality
4. **🧠 Principal's Challenge** - Optional rewrite of a key function

### 2. Create an Issue-to-Location Map

**Before writing any code**, create an explicit mapping using `grep_search` or `view_file`:

| Issue ID | File | Line Range | Function/Struct |
|----------|------|------------|-----------------|
| CB-1 | `src/foo.rs` | L45-L52 | `process_order` |
| MC-1 | `src/bar.rs` | L120-L135 | `BarManager::new` |

**Only files and line ranges in this table should be modified.**

### 3. Create an Implementation Plan

Create or update `implementation_plan.md` with:

- The issue-to-location mapping table from Step 2
- A section for each issue category
- For each issue: ID, description, **exact file:line location**, and proposed fix
- Verification steps for each fix
- **Explicitly state what files WILL NOT be touched**
- Ask the user for approval before proceeding

### 4. Fix Critical Blockers First

For each CB issue:

1. Mark the task as in-progress in `task.md`
2. Use `view_file` to confirm the exact lines to modify
3. Implement the fix using **targeted edits** (prefer `replace_file_content` over `write_to_file`)
4. **Review the diff** - verify changes are within the expected line range
5. Verify the fix compiles: `cargo check`
6. Mark as complete and move to the next

### 5. Fix Major Concerns

For each MC issue:

1. Mark the task as in-progress in `task.md`
2. Use `view_file` to confirm the exact lines to modify
3. Implement the refactoring using **targeted edits**
4. **Review the diff** - verify changes are within the expected line range
5. Verify the fix compiles: `cargo check`
6. Mark as complete and move to the next

### 6. Fix Nitpicks

For each N issue:

1. Mark the task as in-progress in `task.md`
2. Implement the improvement using **targeted edits**
3. **Review the diff** - verify changes match the issue scope
4. Mark as complete and move to the next

### 7. Apply Principal's Challenge (Optional)

If the review included a rewritten function:

1. Review the proposed rewrite
2. Ask the user if they want to apply it
3. If approved, implement the rewrite
4. Document the rationale in code comments

### 8. Verify Change Scope

Before running the verification suite, confirm all changes are in scope:

// turbo
```bash
git diff --stat HEAD~1
```

- **Check**: Only files from the issue-to-location map should appear
- **Check**: Line counts should be reasonable for the fixes applied
- If unexpected files appear, **STOP** and investigate before proceeding

### 9. Run Verification Suite

// turbo
```bash
cargo fmt --all
```

// turbo
```bash
cargo clippy --all-targets --all-features -- -D warnings
```

// turbo
```bash
cargo test
```

### 10. Create Walkthrough

Update `walkthrough.md` with:

- Summary of all fixes applied with file:line references
- Before/after code snippets for significant changes
- Verification evidence (test output, clippy passing)
- **Explicit confirmation that no unrelated code was modified**

### 11. Commit and Push

Follow `/commit_and_push` workflow with a message like:

```
fix: address code review findings for [module name]

- CB-X: [brief description]
- MC-X: [brief description]
- N-X: [brief description]
```

## Scope Control Guidelines

> [!CAUTION]
> **Never modify code outside the identified issue locations without explicit user approval.**

- **Prefer `replace_file_content`** over `write_to_file` to minimize unintended changes
- **Use `multi_replace_file_content`** for multiple non-adjacent edits in the same file
- **Always view the target lines first** before editing
- **Review diffs after each fix** to catch scope creep
- If a fix requires changes to other files (e.g., shared types), **ask the user first**

## Notes

- **Never skip Critical Blockers** - they represent safety violations or UB
- **Ask before applying Principal's Challenge** - it may be opinionated
- **Run tests after each major fix** to catch regressions early
- If a fix reveals additional issues, add them to the task list **but ask the user before fixing**
