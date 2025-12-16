# =============================================================================
# Terraform Configuration for Algopioneer Trading Infrastructure
# =============================================================================

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# --- Variables ---
variable "aws_region" {
  type        = string
  default     = "us-east-1"
  description = "AWS region for deployment"
}

variable "my_ip" {
  type        = string
  description = "Your IP in CIDR notation for SSH access (e.g., 203.0.113.10/32)"
  # No default — force explicit input for security
}

variable "key_name" {
  type        = string
  description = "Name of the EC2 key pair for SSH access"
}

variable "instance_type" {
  type        = string
  default     = "t3.micro"
  description = "EC2 instance type. Override for production (e.g., t3.medium)."
}

# --- Locals for Consistent Naming ---
locals {
  name_prefix = "algopioneer"
  common_tags = {
    Project   = "algopioneer"
    ManagedBy = "terraform"
  }
}

# --- Data Sources ---
data "aws_ami" "amazon_linux_2023" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-*-kernel-6.1-x86_64"]
  }
}

# --- 1. Networking (The Bunker) ---
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-vpc"
  })
}

resource "aws_internet_gateway" "gw" {
  vpc_id = aws_vpc.main.id

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-igw"
  })
}

resource "aws_subnet" "public" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = "${var.aws_region}a"
  map_public_ip_on_launch = true

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-public-subnet"
  })
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.gw.id
  }

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-public-rt"
  })
}

resource "aws_route_table_association" "public" {
  subnet_id      = aws_subnet.public.id
  route_table_id = aws_route_table.public.id
}

# --- 2. Security (The Firewall) ---
resource "aws_security_group" "bot_sg" {
  name        = "${local.name_prefix}-sg"
  description = "Allow SSH from specified IP and outbound to world"
  vpc_id      = aws_vpc.main.id

  # INBOUND: SSH (22) - Restricted to specified IP
  ingress {
    description = "SSH access from admin IP"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.my_ip]
  }

  # OUTBOUND: Allow all (required for Coinbase/Kraken API and ECR)
  # NOTE: Open egress is intentional for exchange API connectivity
  egress {
    description = "All outbound traffic"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-sg"
  })
}

# --- 3. IAM Permissions (Access to ECR) ---
resource "aws_iam_role" "ec2_role" {
  name = "${local.name_prefix}-ec2-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ec2.amazonaws.com"
      }
    }]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "ecr_read" {
  role       = aws_iam_role.ec2_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

resource "aws_iam_instance_profile" "ec2_profile" {
  name = "${local.name_prefix}-ec2-profile"
  role = aws_iam_role.ec2_role.name

  tags = local.common_tags
}

# --- 4. The Server (EC2) ---
resource "aws_instance" "bot" {
  ami                    = data.aws_ami.amazon_linux_2023.id
  instance_type          = var.instance_type
  subnet_id              = aws_subnet.public.id
  vpc_security_group_ids = [aws_security_group.bot_sg.id]
  iam_instance_profile   = aws_iam_instance_profile.ec2_profile.name
  key_name               = var.key_name

  # Observability: Enable detailed CloudWatch monitoring
  monitoring = true

  # Immutable infrastructure: Replace instance on user_data change
  user_data_replace_on_change = true

  # Bootstrap script with logging and error handling
  user_data = <<-EOF
    #!/bin/bash
    set -euxo pipefail
    exec > >(tee /var/log/user-data.log | logger -t user-data) 2>&1

    echo "--- Bootstrapping ${local.name_prefix} trading instance ---"
    dnf update -y
    dnf install -y docker
    systemctl enable --now docker
    usermod -a -G docker ec2-user
    echo "--- Bootstrap complete at $(date) ---"
  EOF

  tags = merge(local.common_tags, {
    Name        = "${local.name_prefix}-shadow"
    Environment = "production"
  })

  lifecycle {
    # Prevent accidental destruction of the trading instance
    prevent_destroy = true
  }
}

# --- Outputs ---
output "public_ip" {
  description = "Public IP address of the trading instance"
  value       = aws_instance.bot.public_ip
}

output "instance_id" {
  description = "EC2 instance ID for reference"
  value       = aws_instance.bot.id
}

output "ami_id" {
  description = "AMI ID used for the instance (for debugging)"
  value       = data.aws_ami.amazon_linux_2023.id
}