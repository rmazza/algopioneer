# --- GitHub Actions OIDC Integration ---

data "aws_caller_identity" "current" {}

# Create the OIDC Provider for GitHub Actions
resource "aws_iam_openid_connect_provider" "github" {
  url             = "https://token.actions.githubusercontent.com"
  client_id_list  = ["sts.amazonaws.com"]
  
  # Thumbprints for token.actions.githubusercontent.com
  thumbprint_list = [
    "1b511abead59c6ce207077c0bf0e0043b1382612",
    "6938fd4d98bab03faadb97b34396831e3780aea1",
    "1c58a3a8518e8759bf075b76b750d4f2df264fcd"
  ]
}

# Create the IAM Role assumed by GitHub Actions
resource "aws_iam_role" "github_actions_release" {
  name = "github-actions-algopioneer-release"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Federated = aws_iam_openid_connect_provider.github.arn
        }
        Action = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          StringEquals = {
            "token.actions.githubusercontent.com:aud" = "sts.amazonaws.com"
          }
          StringLike = {
            # Ensure only the rmazza/algopioneer repository can assume this role
            "token.actions.githubusercontent.com:sub" = "repo:rmazza/algopioneer:*"
          }
        }
      }
    ]
  })
}

# Inline policy to grant ECR Push access to the role
resource "aws_iam_role_policy" "github_actions_ecr_push" {
  name = "github-actions-ecr-push"
  role = aws_iam_role.github_actions_release.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:GetRepositoryPolicy",
          "ecr:DescribeRepositories",
          "ecr:ListImages",
          "ecr:DescribeImages",
          "ecr:BatchGetImage",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload",
          "ecr:PutImage"
        ]
        Resource = "arn:aws:ecr:${var.aws_region}:${data.aws_caller_identity.current.account_id}:repository/algopioneer"
      }
    ]
  })
}
