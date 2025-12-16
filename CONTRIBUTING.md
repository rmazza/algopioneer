# Contributing to AlgoPioneer

## Release Branching Strategy

We use a **Git Flow** branching model to manage releases and features. This ensures a stable production environment while allowing for continuous development.

### Branches

| Branch | Description | Source | Merge Target |
| :--- | :--- | :--- | :--- |
| `main` | Production-ready code. Each commit reflects a production release. | `release/*`, `hotfix/*` | N/A |
| `develop` | Integration branch for the next release. Contains the latest delivered features. | `main` | `main` |
| `feature/*` | New features or non-critical fixes. | `develop` | `develop` |
| `release/v*` | Preparation for a new production release. | `develop` | `main` and `develop` |
| `hotfix/*` | Critical fixes for the live production version. | `main` | `main` and `develop` |

### Workflows

#### 1. Feature Development
*   **Start**: Create a branch off `develop`: `git checkout -b feature/my-feature develop`
*   **Work**: Commit your changes.
*   **Finish**: Open a Pull Request (PR) to merge `feature/my-feature` into `develop`.

#### 2. Release Process
*   **Start**: When `develop` is ready for a release, create a release branch: `git checkout -b release/v1.0.0 develop`
*   **Work**: Perform final bug fixes, documentation updates, and version bumping. **No new features.**
*   **Finish**: 
    1.  Merge `release/v1.0.0` into `main`. Tag this commit (e.g., `v1.0.0`).
    2.  Merge `release/v1.0.0` back into `develop`.
    3.  Delete the release branch.

#### 3. Hotfix Process
*   **Start**: If a critical bug is found in production, branch off `main`: `git checkout -b hotfix/critical-bug main`
*   **Work**: Fix the bug.
*   **Finish**:
    1.  Merge `hotfix/critical-bug` into `main`. Tag this commit (e.g., `v1.0.1`).
    2.  Merge `hotfix/critical-bug` into `develop`.
    3.  Delete the hotfix branch.

### Versioning
We follow [Semantic Versioning](https://semver.org/) (`vMajor.Minor.Patch`).

---
