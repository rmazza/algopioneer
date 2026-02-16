#!/bin/bash
set -euxo pipefail

# 1. Fetch Secrets from SSM
echo "Fetching secrets from SSM..."
export ALPACA_API_KEY=$(aws ssm get-parameter --name "/algopioneer/alpaca/api-key" --with-decryption --query "Parameter.Value" --output text --region us-east-1)
export ALPACA_API_SECRET=$(aws ssm get-parameter --name "/algopioneer/alpaca/api-secret" --with-decryption --query "Parameter.Value" --output text --region us-east-1)

if [ -z "$ALPACA_API_KEY" ] || [ -z "$ALPACA_API_SECRET" ]; then
  echo "Error: Failed to fetch secrets"
  exit 1
fi

# 2. ECR Login & Pull
echo "Logging into ECR..."
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 756471705399.dkr.ecr.us-east-1.amazonaws.com

# echo "Pulling image..."
# docker pull 756471705399.dkr.ecr.us-east-1.amazonaws.com/algopioneer:v1.4.0

# 3. Stop/Clean existing container
echo "Stopping old container..."
docker stop algopioneer-alpaca || true
docker rm algopioneer-alpaca || true

# 4. Run Container
echo "Starting new container..."
docker run -d \
  --name algopioneer-alpaca \
  --restart unless-stopped \
  --log-driver=awslogs \
  --log-opt awslogs-region=us-east-1 \
  --log-opt awslogs-group=/algopioneer/alpaca-paper \
  --log-opt awslogs-create-group=true \
  --log-opt awslogs-stream=alpaca-$(date +%Y%m%d-%H%M%S) \
  -e ALPACA_API_KEY="$ALPACA_API_KEY" \
  -e ALPACA_API_SECRET="$ALPACA_API_SECRET" \
  -v /home/ec2-user/discovered_pairs.json:/app/discovered_pairs.json:ro \
  -v /home/ec2-user/paper_trades_alpaca.csv:/app/paper_trades_alpaca.csv \
  algopioneer:v1.7.4 \
  portfolio \
  --config /app/discovered_pairs.json \
  --exchange alpaca \
  --paper

echo "Deployment complete. Logs available in CloudWatch group /algopioneer/alpaca-paper"
docker ps | grep algopioneer
