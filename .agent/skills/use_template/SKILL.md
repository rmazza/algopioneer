---
name: Use Template
description: Scaffold new files or artifacts using standard templates.
---

# Use Template

This skill helps you generate new files (issues, PRs, docs, strategies) using standardized templates from the `resources/` directory.

## Instructions

### 1. Select Template
Identify the needed template.
*   `issue_template.md`
*   `pr_description.md`
*   `changelog_entry.md`
*   ...

### 2. Read Template
Load the content from `resources/[template_name].md`.

### 3. Fill Content
*   **System Fill**: If you have the context (e.g., git branch name, recent commits), fill in the template automatically.
*   **User Interview**: If information is missing, ask the user (e.g., "What is the risk level?").

### 4. Output
Write the populated content to the target file or present it to the user.
