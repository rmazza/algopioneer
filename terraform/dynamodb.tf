# DynamoDB Table for Trade Recording
resource "aws_dynamodb_table" "trades" {
  name         = "algopioneer-trades"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "pk"
  range_key    = "sk"

  attribute {
    name = "pk"
    type = "S"
  }

  attribute {
    name = "sk"
    type = "S"
  }

  tags = merge(local.common_tags, {
    Name = "algopioneer-trades"
  })
}
