---
name: Deploy Application
description: Unified deployment skill for Application (Docker) and Infrastructure (Terraform).
---

# Deploy Application

This skill handles all deployment activities, whether updating the application code or the underlying infrastructure.

## Sub-Routines

### 1. Application Deploy (Local Build -> EC2)
*   **Use when**: Updating code, strategy logic, or config.
*   **Process**:
    1.  **Prereq**: Get EC2 IP, check git status.
    2.  **Build**: `docker build --platform linux/amd64 ...`
    3.  **Ship**: `docker save ... | ssh ... docker load`
    4.  **Restart**: Stop old container, start new one with correct flags.
    5.  **Verify**: Check `docker ps` and logs.

### 2. Infrastructure Deploy (Terraform)
*   **Use when**: Changing AWS resources (EC2 instance type, IAM roles, Security Groups).
*   **Process**:
    1.  **Init**: `terraform init` (in `terraform/` dir).
    2.  **Plan**: `terraform plan -out=tfplan` -> **Critical User Review**.
    3.  **Apply**: `terraform apply tfplan`
    4.  **Output**: Show new resource details (e.g., new IP).
