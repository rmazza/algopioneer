---
description: Manage infrastructure deployment with Terraform
---

# Deploy Infrastructure

This workflow guides you through the process of updating and deploying infrastructure using Terraform.

## Prerequisites

- `terraform` CLI installed.
- AWS credentials configured (if deploying to AWS).
- Navigate to the `terraform` directory: `cd terraform`.

## Steps

### 1. Initialize Terraform
// turbo
Initialize the working directory, downloading plugin libraries and modules.

```bash
terraform init
```

### 2. Select Workspace/Environment
Check which environment you are targeting.

```bash
terraform workspace list
# To switch: terraform workspace select <env>
```

### 3. Plan Changes
Generate an execution plan to see what Terraform will do. always review this carefully.

```bash
terraform plan -out=tfplan
```

### 4. Apply Changes
**Action Required**: Review the plan output from the previous step. only proceed if the changes are intended.

```bash
terraform apply tfplan
```

### 5. Verify Output
Check the output variables (e.g., EC2 public IP) to ensure the deployment was successful.

```bash
terraform output
```
