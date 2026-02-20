# Pull Request Template

## Summary
<!-- Brief description of the changes -->

## Type of Change
- [ ] 🐛 Bug fix (non-breaking change that fixes an issue)
- [ ] ✨ New feature (non-breaking change that adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to change)
- [ ] 📝 Documentation update
- [ ] 🔧 Refactoring (no functional changes)
- [ ] 🚀 Performance improvement

## Changes Made
<!-- List the specific changes -->
- 
- 
- 

## Testing
<!-- Describe how you tested these changes -->
- [ ] Unit tests pass (`cargo test`)
- [ ] Clippy clean (`cargo clippy --all-targets --all-features -- -D warnings`)
- [ ] Formatting verified (`cargo fmt --all -- --check`)
- [ ] Dependency check (`cargo deny check`)

## Trading-Specific Checks
<!-- Complete if changes affect trading logic -->
- [ ] All financial calculations use `rust_decimal::Decimal`
- [ ] No `unwrap()` or `expect()` in order execution paths
- [ ] Circuit breaker coverage for new failure modes
- [ ] Paper trading tested before live consideration

## Risk Assessment
<!-- Impact if something goes wrong -->
- **Risk Level**: Low / Medium / High
- **Rollback Plan**: 

## Related Issues
<!-- Link to related issues -->
Closes #

## Screenshots/Logs
<!-- If applicable, add screenshots or log snippets -->
