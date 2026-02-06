# =============================================================================
# Terraform Monitoring Configuration
# =============================================================================
# Defines CloudWatch Alarms and SNS Topics for operational alerts.

# --- 1. Notification Channel (SNS) ---
resource "aws_sns_topic" "alerts" {
  name = "${local.name_prefix}-alerts"

  tags = local.common_tags
}

# --- 2. System Health Alarms ---

# Alarm: Underlying EC2 hardware failure or OS crash
resource "aws_cloudwatch_metric_alarm" "ec2_status_check_failed" {
  alarm_name          = "${local.name_prefix}-ec2-status-check-failed"
  comparison_operator = "GreaterThanOrEqualToThreshold"
  evaluation_periods  = "2"
  metric_name         = "StatusCheckFailed"
  namespace           = "AWS/EC2"
  period              = "60"
  statistic           = "Maximum"
  threshold           = "1"
  alarm_description   = "Triggered when EC2 status checks fail (system or instance reachable)"
  actions_enabled     = true

  dimensions = {
    InstanceId = aws_instance.bot.id
  }

  alarm_actions = [aws_sns_topic.alerts.arn]
  ok_actions    = [aws_sns_topic.alerts.arn]
  tags          = local.common_tags
}

# Alarm: High CPU usage (potential runaway process / infinite loop)
resource "aws_cloudwatch_metric_alarm" "ec2_cpu_high" {
  alarm_name          = "${local.name_prefix}-ec2-cpu-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = "300"
  statistic           = "Average"
  threshold           = "80"
  alarm_description   = "Triggered when CPU > 80% for 10 minutes"
  actions_enabled     = true

  dimensions = {
    InstanceId = aws_instance.bot.id
  }

  alarm_actions = [aws_sns_topic.alerts.arn]
  ok_actions    = [aws_sns_topic.alerts.arn]
  tags          = local.common_tags
}

# --- Outputs ---
output "sns_topic_arn" {
  description = "ARN of the SNS alert topic"
  value       = aws_sns_topic.alerts.arn
}
