#!/bin/bash
set -e

# Configuration
EC2_IP="3.239.222.18"
KEY_PATH="~/.ssh/trading-key.pem"
REMOTE_USER="ec2-user"
REMOTE_DIR="~/algopioneer"

echo "=== Deploying Alpaca Fix to EC2 ($EC2_IP) ==="

# 1. Update Code
echo "--> Updating code..."
ssh -o StrictHostKeyChecking=no -i $KEY_PATH $REMOTE_USER@$EC2_IP "cd $REMOTE_DIR && git pull origin main"

# 2. Rebuild Image (Since we changed Rust code, we must rebuild)
echo "--> Rebuilding Docker image (this may take a few minutes)..."
ssh -o StrictHostKeyChecking=no -i $KEY_PATH $REMOTE_USER@$EC2_IP "cd $REMOTE_DIR && docker build -t algopioneer:latest ."

# 3. Restart Container
echo "--> Restarting algopioneer-alpaca container..."
ssh -o StrictHostKeyChecking=no -i $KEY_PATH $REMOTE_USER@$EC2_IP "
  docker stop algopioneer-alpaca || true 
  docker rm algopioneer-alpaca || true
  
  # Fetch secrets (assuming they are in .env or environment)
  # We reuse the existing run command logic but ensure we use the new image
  
  # Note: The previous run command used v1.4.0 tag, we will use 'latest' for this hotfix
  # or we should tag it. Let's use 'latest' for immediate verification.
  
  # Re-run with same parameters as before
  docker run -d \\
    --name algopioneer-alpaca \\
    --restart unless-stopped \\
    --env-file .env \\
    -v \$(pwd)/discovered_pairs_730d.json:/app/pairs.json \\
    algopioneer:latest \\
    portfolio \\
    --exchange alpaca \\
    --paper \\
    --config pairs.json
"

echo "=== Deployment Complete ==="
echo "Verifying logs..."
ssh -o StrictHostKeyChecking=no -i $KEY_PATH $REMOTE_USER@$EC2_IP "docker logs --tail 20 algopioneer-alpaca"
