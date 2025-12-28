---
description: Run the full suite of integrity checks (fmt, clippy, test, deny)
---

# Verify Integrity

This workflow runs the standard suite of verification checks that must pass before any code is committed. It adheres to the "Five Pillars of Integrity" by ensuring code style, linting, correctness, and dependency compliance.

## Steps

### 1. Check Formatting
// turbo
Ensure all code satisfies the standard Rust style guide:

```bash
cargo fmt --all -- --check
```

### 2. Run Linter
// turbo
Run Clippy to catch common mistakes and improve code quality. Treating warnings as errors ensures a clean codebase:

```bash
cargo clippy --all-targets --all-features -- -D warnings
```

### 3. Run Tests
// turbo
Execute the unit and integration test suite:

```bash
cargo test
```

### 4. Check Dependencies
// turbo
Verify that all dependencies satisfy the `cargo-deny` policy (licenses, advisories, bans, sources):

```bash
cargo deny check
```

## Troubleshooting

- **Formatting failures**: Run `cargo fmt` to automatically fix style issues.
- **Clippy failures**: Follow the suggestions provided by the compiler.
- **Test failures**: Inspect the output to pinpoint the failing test case.
- **Deny failures**: Check `deny.toml` or update dependencies if there are security advisories.
