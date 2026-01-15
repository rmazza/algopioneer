---
description: Build Docker image locally and deploy directly to EC2
---

# Local Build and EC2 Deploy

Build the Docker image on your local machine and push it directly to EC2, bypassing CI/CD. Use when GitHub Actions is unavailable or for rapid iteration.

// turbo-all

## Prerequisites

```bash
# Get EC2 IP
EC2_IP=$(cd terraform && terraform output -raw public_ip 2>/dev/null) || \
EC2_IP=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=*algopioneer*" "Name=instance-state-name,Values=running" \
  --query "Reservations[0].Instances[0].PublicIpAddress" --output text)
echo "EC2 IP: $EC2_IP"
```

> [!IMPORTANT]
> Ensure Docker is running locally and you have SSH access to the EC2 instance.

---

## Steps

### 0. Review Pre-Deploy Checklist

Before deploying, review the pre-deploy checklist:

```bash
cat .agent/checklists/pre_deploy.md
```

> [!CAUTION]
> Confirm `--exchange alpaca` is set (defaults to coinbase otherwise).

---

### 1. Clean Working Directory

Verify git status is clean:

```bash
git status --short
```

---

### 2. Build Docker Image Locally

Build for linux/amd64 (EC2 architecture):

```bash
docker build --platform linux/amd64 -t algopioneer:latest .
```

> [!NOTE]
> This takes 5-10 minutes on first build (cargo-chef caches dependencies).

---

### 3. Save and Transfer Image

Export image and copy to EC2:

```bash
# Save image to tarball
docker save algopioneer:latest | gzip > /tmp/algopioneer.tar.gz

# Copy to EC2
scp -i ~/.ssh/trading-key.pem /tmp/algopioneer.tar.gz ec2-user@$EC2_IP:/tmp/
```

---

### 4. Load Image on EC2

SSH into EC2 and load the image:

```bash
ssh -i ~/.ssh/trading-key.pem ec2-user@$EC2_IP 'gunzip -c /tmp/algopioneer.tar.gz | docker load'
```

---

### 5. Stop Old Container

Stop the currently running Alpaca container:

```bash
ssh -i ~/.ssh/trading-key.pem ec2-user@$EC2_IP 'docker stop algopioneer-alpaca 2>/dev/null; docker rm algopioneer-alpaca 2>/dev/null; echo "Old container removed"'
```

---

### 6. Deploy New Container

Start the new container with the Alpaca pairs config:

```bash
ssh -i ~/.ssh/trading-key.pem ec2-user@$EC2_IP 'docker run -d \
  --name algopioneer-alpaca \
  --restart unless-stopped \
  --env-file /home/ec2-user/.env \
  -v /home/ec2-user/discovered_pairs.json:/app/discovered_pairs.json:ro \
  -v /home/ec2-user/paper_trades_alpaca.csv:/app/paper_trades_alpaca.csv \
  -p 8080:8080 \
  algopioneer:latest \
  portfolio --config /app/discovered_pairs.json --exchange alpaca --paper'
```

> [!WARNING]
> The `--exchange alpaca` flag is **required**. Without it, the application defaults to Coinbase and will fail if Coinbase credentials are not present.

---

### 7. Verify Deployment

Check container is running and healthy:

```bash
ssh -i ~/.ssh/trading-key.pem ec2-user@$EC2_IP 'docker ps | grep algopioneer-alpaca && docker logs --tail 20 algopioneer-alpaca'
```

---

### 8. Cleanup Local Tarball

Remove the temporary tarball:

```bash
rm -f /tmp/algopioneer.tar.gz
```

---

## Quick Deploy (One-Liner)

For subsequent deploys after initial setup:

```bash
EC2_IP=$(cd terraform && terraform output -raw public_ip) && \
docker build --platform linux/amd64 -t algopioneer:latest . && \
docker save algopioneer:latest | gzip | ssh -i ~/.ssh/trading-key.pem ec2-user@$EC2_IP 'gunzip | docker load' && \
ssh -i ~/.ssh/trading-key.pem ec2-user@$EC2_IP 'docker stop algopioneer-alpaca; docker rm algopioneer-alpaca; docker run -d --name algopioneer-alpaca --restart unless-stopped --env-file ~/.env -v ~/discovered_pairs.json:/app/discovered_pairs.json:ro -v ~/paper_trades_alpaca.csv:/app/paper_trades_alpaca.csv -p 8080:8080 algopioneer:latest portfolio --config /app/discovered_pairs.json --exchange alpaca --paper'
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `docker: command not found` on EC2 | `sudo yum install -y docker && sudo systemctl start docker` |
| Permission denied on docker | `sudo usermod -aG docker ec2-user` and re-login |
| Image too large | Use `docker system prune` before building |
| SSH timeout | Check security group allows port 22 from your IP |
