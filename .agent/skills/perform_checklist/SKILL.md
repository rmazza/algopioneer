---
name: Perform Checklist
description: Interactively run any checklist from the resources directory.
---

# Perform Checklist

This skill allows you to select and interactively perform a checklist. This ensures consistency and prevents skipped steps in critical processes.

## Instructions

### 1. Select Checklist
List available checklists in `resources/` and ask the user which one to perform if not specified.
*   `pre_deploy.md`
*   `security.md`
*   `code_quality.md`
*   ...

### 2. Load Checklist
Read the content of the selected checklist.

### 3. Execute Item-by-Item
Iterate through the checklist items:
1.  **State the Item**: "Checking: [Item Description]"
2.  **Verify**: Perform the check (if automated) or ask the user for confirmation.
3.  **Mark**: Record the status (Pass/Fail/Skip).

### 4. Report
At the end, provide a summary:
*   **Passed**: [Count]
*   **Failed**: [Count] (List failures)
*   **Skipped**: [Count]

> [!IMPORTANT]
> If a critical item fails (e.g., "Security Audit"), simpler checklists may proceed, but deployment checklists should generally halt.
