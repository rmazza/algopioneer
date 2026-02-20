---
name: Release Flow
description: Perform a production release using Git Flow (bump version, changelog, tag, merge).
---

# Release Flow

This skill manages the production release process, ensuring Semantic Versioning and Git Flow compliance.

## Prerequisites

*   Clean git working directory on `develop` branch.
*   Passing tests (`cargo test`).

## Instructions

### 1. Determine Version
Analyze commit history since the last tag.
```bash
git log $(git describe --tags --abbrev=0)..HEAD --oneline
```
*   **Major (X.0.0)**: Breaking API changes.
*   **Minor (x.Y.0)**: New features (backward compatible).
*   **Patch (x.y.Z)**: Bug fixes.

### 2. Create Release Branch
Create a branch for the new version.
```bash
git checkout -b release/vX.Y.Z develop
```

### 3. Update Artifacts
1.  **Cargo.toml**: Update `version = "X.Y.Z"`.
2.  **CHANGELOG.md**: Add a section `[X.Y.Z] - YYYY-MM-DD` and move "Unreleased" items there.

### 4. Verify
Run the full test suite to ensure the release candidate is stable.
```bash
cargo test
```

### 5. Commit & Merge
1.  Commit changes: `git commit -am "chore: bump version to vX.Y.Z"`
2.  **Merge to Main**:
    ```bash
    git checkout main
    git pull origin main
    git merge release/vX.Y.Z
    ```
3.  **Tag**:
    ```bash
    git tag -a vX.Y.Z -m "Release vX.Y.Z"
    git push origin main --tags
    ```
4.  **Merge Back to Develop**:
    ```bash
    git checkout develop
    git merge release/vX.Y.Z
    ```

### 6. Cleanup
Delete the release branch.
```bash
git branch -d release/vX.Y.Z
git push origin develop
```

### 7. Monitor CI
Watch the GitHub Actions / Build pipeline to ensure the Docker image is built and pushed to ECR.
