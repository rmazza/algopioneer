---
description: Fix issues identified by the /code_review workflow
---

This workflow is used **after** running `/code_review` on a file. It systematically addresses and fixes all identified issues.

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

### 2. Create an Implementation Plan

Create or update `implementation_plan.md` with:

- A section for each issue category
- For each issue: ID, description, file location, and proposed fix
- Verification steps for each fix
- Ask the user for approval before proceeding

### 3. Fix Critical Blockers First

For each CB issue:

1. Mark the task as in-progress in `task.md`
2. Implement the fix following the plan
3. Verify the fix compiles: `cargo check`
4. Mark as complete and move to the next

### 4. Fix Major Concerns

For each MC issue:

1. Mark the task as in-progress in `task.md`
2. Implement the refactoring following the plan
3. Verify the fix compiles: `cargo check`
4. Mark as complete and move to the next

### 5. Fix Nitpicks

For each N issue:

1. Mark the task as in-progress in `task.md`
2. Implement the improvement
3. Mark as complete and move to the next

### 6. Apply Principal's Challenge (Optional)

If the review included a rewritten function:

1. Review the proposed rewrite
2. Ask the user if they want to apply it
3. If approved, implement the rewrite
4. Document the rationale in code comments

### 7. Run Verification Suite

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

### 8. Create Walkthrough

Update `walkthrough.md` with:

- Summary of all fixes applied
- Before/after code snippets for significant changes
- Verification evidence (test output, clippy passing)

### 9. Commit and Push

Follow `/commit_and_push` workflow with a message like:

```
fix: address code review findings for [module name]

- CB-X: [brief description]
- MC-X: [brief description]
- N-X: [brief description]
```

## Notes

- **Never skip Critical Blockers** - they represent safety violations or UB
- **Ask before applying Principal's Challenge** - it may be opinionated
- **Run tests after each major fix** to catch regressions early
- If a fix reveals additional issues, add them to the task list
