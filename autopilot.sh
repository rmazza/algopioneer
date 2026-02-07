#!/bin/bash
set -eo pipefail

# === Configuration ===
APP_DIR="/home/ec2-user"
CURRENT_CONFIG="$APP_DIR/discovered_pairs.json"
NEW_CONFIG="$APP_DIR/discovered_pairs_new.json"
COMPARE_SCRIPT="$APP_DIR/compare_pairs.py"
DEPLOY_SCRIPT="$APP_DIR/deploy_alpaca.sh"
LOG_FILE="$APP_DIR/autopilot.log"
IMAGE="756471705399.dkr.ecr.us-east-1.amazonaws.com/algopioneer:v1.7.0"

# Helper for logging
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "=== Starting Autopilot Health Check ==="

# 1. Fetch AWS Secrets (needed for Discovery)
# We need to export them for the docker run command
log "Fetching secrets..."
ALPACA_API_KEY=$(aws ssm get-parameter --name "/algopioneer/alpaca/api-key" --with-decryption --query "Parameter.Value" --output text --region us-east-1)
ALPACA_API_SECRET=$(aws ssm get-parameter --name "/algopioneer/alpaca/api-secret" --with-decryption --query "Parameter.Value" --output text --region us-east-1)

if [ -z "$ALPACA_API_KEY" ]; then
    log "ERROR: Could not fetch secrets."
    exit 1
fi

# 2. Run Discovery (Disposable Container)
log "Running Pair Discovery..."
# We use the same image as production. 
# We mount the directory to save the output.
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 756471705399.dkr.ecr.us-east-1.amazonaws.com

docker run --rm \
  -e ALPACA_API_KEY="$ALPACA_API_KEY" \
  -e ALPACA_API_SECRET="$ALPACA_API_SECRET" \
  -v "$APP_DIR":/app/output \
  "$IMAGE" \
  discover-pairs \
  --exchange alpaca \
  --symbols default \
  --lookback-days 730 \
  --initial-capital 2000 \
  --output /app/output/discovered_pairs_new.json

if [ ! -f "$NEW_CONFIG" ]; then
    log "ERROR: Discovery failed to produce output file."
    exit 1
fi

# 3. Compare Results
log "Comparing new pairs with current configuration..."
if python3 "$COMPARE_SCRIPT" "$CURRENT_CONFIG" "$NEW_CONFIG" >> "$LOG_FILE"; then
    log "No changes recommended. Autopilot finished."
    rm -f "$NEW_CONFIG"
    exit 0
else
    # 4. Redeploy needed (Exit code 1 from python script)
    log "CHANGES DETECTED. Initiating Redeployment..."
    
    # Backup
    cp "$CURRENT_CONFIG" "${CURRENT_CONFIG}.bak_$(date +%Y%m%d_%H%M%S)"
    
    # Update Config
    mv "$NEW_CONFIG" "$CURRENT_CONFIG"
    
    # Run Deploy Script
    if bash "$DEPLOY_SCRIPT" >> "$LOG_FILE" 2>&1; then
        log "Redeployment Successful."
    else
        log "ERROR: Redeployment Failed!"
        exit 1
    fi
fi
