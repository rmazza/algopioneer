---
description: Perform a production release using Git Flow
---

This workflow guides you through the process of creating a production release, from cutting the branch to merging and tagging.

1.  **Preparation**: Ensure you are on the `develop` branch and have a clean working directory.
    ```bash
    git checkout develop
    git pull origin develop
    ```

    > [!IMPORTANT]
    > Ensure that `git status` is clean before proceeding.

2.  **Determine Version**: Review changes to decide on the next version.
    Check the commit history since the last release (or start of project if no tags exist).
    ```bash
    # If you have tags: git log $(git describe --tags --abbrev=0)..HEAD --oneline
    # If no tags yet:
    git log --oneline
    ```
    
    **Decision Guide**:
    - **Major (X.0.0)**: Did you change an existing API (e.g., rename a public function)?
    - **Minor (x.Y.0)**: Did you add a new feature (e.g., new strategy command)?
    - **Patch (x.y.Z)**: Did you fix a bug without changing features?

3.  **Input Version**: Set the version for the new release.
    > [!NOTE]
    > Replace `vX.Y.Z` with your desired version (e.g., `v1.0.0`) in the commands below.
    > 
    > **Versioning**: This project follows [Semantic Versioning](https://semver.org/):
    > - **Major (1.0.0)**: Breaking API changes
    > - **Minor (0.2.0)**: New features (backward compatible)
    > - **Patch (0.0.3)**: Bug fixes (backward compatible)

4.  **Create Release Branch**:
    ```bash
    # Replace vX.Y.Z with the new version (e.g. v1.0.0)
    git checkout -b release/vX.Y.Z develop
    ```

5.  **Bump Version**: Update the version in `Cargo.toml`.
    - Open `Cargo.toml`.
    - Update `version = "..."` to the new version `X.Y.Z`.
    - Save the file.
    *(You can use `sed` if you are confident, but manual is safer to avoid accidents).*

6.  **Running Tests**: Ensure the release is stable (This step runs automatically).
    // turbo
    ```bash
    cargo test
    ```

7.  **Commit Changes**:
    ```bash
    git add Cargo.toml
    git commit -m "chore: bump version to vX.Y.Z"
    ```

8.  **Merge to Main (Production)**:
    ```bash
    git checkout main
    git pull origin main
    git merge release/vX.Y.Z
    ```

9.  **Tag Release**:
    ```bash
    git tag -a vX.Y.Z -m "Release vX.Y.Z"
    git push origin main --tags
    ```

10. **Merge Back to Develop**:
    ```bash
    git checkout develop
    git merge release/vX.Y.Z
    ```

11. **Cleanup**:
    ```bash
    git branch -d release/vX.Y.Z
    git push origin develop
    ```

12. **Wait for CI (GitHub Actions)**:
    The push of the tag `vX.Y.Z` will trigger the release workflow.
    -   **Action**: Builds Docker image.
    -   **Registry**: Pushes to AWS ECR (public).
    
    Check the [Actions tab](https://github.com/your-repo/actions) for progress.
    ```
