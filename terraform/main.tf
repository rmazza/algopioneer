provider "aws" {
  region = "us-east-1"
}

# --- 1. Networking (The Bunker) ---
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
  tags = { Name = "algopioneer-vpc" }
}

resource "aws_internet_gateway" "gw" {
  vpc_id = aws_vpc.main.id
}

resource "aws_subnet" "public" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = "us-east-1a"
  map_public_ip_on_launch = true # Required for SSH access
  tags = { Name = "algopioneer-public-subnet" }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.gw.id
  }
}

resource "aws_route_table_association" "public" {
  subnet_id      = aws_subnet.public.id
  route_table_id = aws_route_table.public.id
}

# --- 2. Security (The Firewall) ---
resource "aws_security_group" "bot_sg" {
  name        = "algopioneer-sg"
  description = "Allow SSH from My IP and Outbound to World"
  vpc_id      = aws_vpc.main.id

  # INBOUND: SSH (22) - REPLACE WITH YOUR IP
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["YOUR_HOME_IP/32"] # <--- CRITICAL: Put your IP here
  }

  # OUTBOUND: Allow Everything (Need to talk to Coinbase/ECR)
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# --- 3. IAM Permissions (Access to ECR) ---
resource "aws_iam_role" "ec2_role" {
  name = "algopioneer-ec2-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ecr_read" {
  role       = aws_iam_role.ec2_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

resource "aws_iam_instance_profile" "ec2_profile" {
  name = "algopioneer-ec2-profile"
  role = aws_iam_role.ec2_role.name
}

# --- 4. The Server (EC2) ---
resource "aws_instance" "bot" {
  ami           = "ami-0c55b159cbfafe1f0" # Amazon Linux 2023 (us-east-1)
  instance_type = "t3.micro"              # Free Tier Eligible
  subnet_id     = aws_subnet.public.id
  
  vpc_security_group_ids = [aws_security_group.bot_sg.id]
  iam_instance_profile   = aws_iam_instance_profile.ec2_profile.name
  key_name               = "trading-key"  # Assumes you created this in AWS Console

  # User Data: Install Docker on boot
  user_data = <<-EOF
              #!/bin/bash
              yum update -y
              yum install -y docker
              service docker start
              usermod -a -G docker ec2-user
              EOF

  tags = { Name = "algopioneer-shadow" }
}

output "public_ip" {
  value = aws_instance.bot.public_ip
}