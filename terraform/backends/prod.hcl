# =============================================================================
# Production Backend Configuration (S3 + DynamoDB Locking)
# =============================================================================
# Usage: terraform init -backend-config=backends/prod.hcl
#
# Prerequisites:
#   1. Create S3 bucket: aws s3 mb s3://algopioneer-terraform-state
#   2. Enable versioning: aws s3api put-bucket-versioning \
#        --bucket algopioneer-terraform-state \
#        --versioning-configuration Status=Enabled
#   3. Create DynamoDB table: aws dynamodb create-table \
#        --table-name algopioneer-terraform-lock \
#        --attribute-definitions AttributeName=LockID,AttributeType=S \
#        --key-schema AttributeName=LockID,KeyType=HASH \
#        --billing-mode PAY_PER_REQUEST

bucket         = "algopioneer-terraform-state"
key            = "prod/terraform.tfstate"
region         = "us-east-1"
encrypt        = true
dynamodb_table = "algopioneer-terraform-lock"
