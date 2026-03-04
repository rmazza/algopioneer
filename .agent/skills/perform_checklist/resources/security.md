# Security Checklist

Security requirements for trading system code.

## Credentials
- [ ] API keys loaded from environment variables
- [ ] No hardcoded secrets in code
- [ ] `.env` is in `.gitignore`
- [ ] No secrets in git history
- [ ] Keys use minimal required permissions

## Network
- [ ] All API calls use HTTPS
- [ ] WebSocket uses WSS
- [ ] TLS verification enabled
- [ ] Timeouts on all network operations
- [ ] Rate limiting implemented

## Logging
- [ ] Credentials never logged
- [ ] PII not logged
- [ ] Error messages don't leak sensitive data
- [ ] Log levels appropriate for production

## Dependencies
- [ ] `cargo deny check` passes
- [ ] No known vulnerabilities
- [ ] Dependencies pinned in `Cargo.lock`
- [ ] Minimal dependency surface

## Runtime
- [ ] Docker container runs as non-root
- [ ] Minimal file permissions
- [ ] Environment variables not exposed
- [ ] Process isolation maintained

## Incident Response
- [ ] Credential rotation procedure documented
- [ ] Emergency shutdown procedure documented
- [ ] Contact information for exchange support
