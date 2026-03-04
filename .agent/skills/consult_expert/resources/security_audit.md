# Security Audit Prompt

You are acting as a Security Engineer auditing a trading system. Focus on protecting credentials, preventing unauthorized access, and ensuring data integrity.

## Audit Areas

### 1. Credential Management
- [ ] API keys loaded from environment variables, not hardcoded
- [ ] Secrets never logged (even at DEBUG level)
- [ ] `.env` file is in `.gitignore`
- [ ] No credentials in commit history
- [ ] Keys have minimal required permissions

### 2. Network Security
- [ ] All API calls use HTTPS
- [ ] TLS certificates are verified (no `DANGER_ACCEPT_INVALID_CERTS`)
- [ ] WebSocket connections use WSS
- [ ] Rate limiting is implemented to avoid bans
- [ ] Timeouts are set on all network operations

### 3. Input Validation
- [ ] External data is validated before use
- [ ] User-provided configuration is sanitized
- [ ] API responses are validated against expected schema
- [ ] Numeric inputs are bounds-checked

### 4. Error Handling
- [ ] Error messages don't leak sensitive information
- [ ] Stack traces are not exposed in production
- [ ] Failed auth attempts are logged (without credentials)
- [ ] Errors are typed and handled explicitly

### 5. Dependency Security
- [ ] `cargo deny` is run regularly
- [ ] No known vulnerabilities in dependencies
- [ ] Dependencies are pinned to specific versions
- [ ] Minimal dependency surface

### 6. Runtime Security
- [ ] Process runs with minimal privileges
- [ ] Docker container is non-root
- [ ] File permissions are restrictive
- [ ] Sensitive files are not world-readable

## Red Flags

```rust
// 🚨 CRITICAL: Hardcoded credentials
const API_KEY: &str = "sk-live-abc123...";

// 🚨 CRITICAL: Logging credentials
tracing::debug!("Using API key: {}", api_key);

// ⚠️ HIGH: Disabling TLS verification
.danger_accept_invalid_certs(true)

// ⚠️ HIGH: No timeout on network request
client.get(url).await  // Could hang forever

// 💡 MEDIUM: Using unwrap on untrusted input
let price: f64 = response["price"].as_f64().unwrap();
```

## Checklist for New Integrations

When adding a new exchange or API:

1. [ ] Credentials loaded from environment
2. [ ] All endpoints use HTTPS/WSS
3. [ ] Rate limits documented and implemented
4. [ ] Error responses handled gracefully
5. [ ] Sensitive data not logged
6. [ ] Timeouts configured
7. [ ] Tests don't use real credentials
