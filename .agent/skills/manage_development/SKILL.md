---
name: Manage Development
description: Unified workflow for coding, testing, documenting, and saving progress.
---

# Manage Development

This skill streamlines the daily developer loop: scaffolding new features, verifying code quality, documenting changes, and saving work.

## Sub-Routines

### 1. Scaffold Feature / Strategy
*   **Goal**: Create new files from templates.
*   **Steps**:
    1.  Identify type (Strategy, Endpoint, Model).
    2.  Generate file with boilerplate (imports, structs, traits).
    3.  Register module in `mod.rs`.

### 2. Verify Integrity (The Gatekeeper)
*   **Goal**: Ensure code is production-ready.
*   **Steps**:
    1.  `cargo fmt --all -- --check`
    2.  `cargo clippy --all-targets -- -D warnings`
    3.  `cargo test`
    4.  (Optional) `cargo audit`

### 3. Update Documentation
*   **Goal**: Keep docs in sync with code.
*   **Steps**:
    1.  `README.md`: Update build instructions or feature lists.
    2.  `GEMINI.md`: Update AI context, architecture, or tech stack.

### 4. Save Progress (Commit & Push)
*   **Goal**: Securely save work.
*   **Steps**:
    1.  `git status` & `git add`
    2.  **Commit**: Use Conventional Commits (`feat:`, `fix:`, `chore:`).
    3.  **Push**: `git push origin <branch>` (with optional rebase).
