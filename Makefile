.PHONY: verify backtest deploy

verify:
	cargo fmt --all -- --check
	cargo clippy --all-targets --all-features -- -D warnings
	cargo test
	cargo deny check

backtest:
	cargo run --release -- backtest --strategy moving-average --symbols BTC-USD --duration 7d

deploy:
	cd terraform && terraform init && terraform plan -out=tfplan && terraform apply tfplan
