use algopioneer::discovery::config::PortfolioPairConfig;
use std::fs::File;
use std::io::BufReader;

fn main() {
    let file = File::open("discovered_pairs.json").expect("Failed to open file");
    let reader = BufReader::new(file);
    let configs: Vec<PortfolioPairConfig> =
        serde_json::from_reader(reader).expect("Failed to parse");

    println!("Successfully parsed {} configs.", configs.len());

    for (i, config) in configs.iter().enumerate() {
        println!(
            "Entry #{}: Symbol={}, StopLoss={}, MinProfit={}",
            i,
            config.dual_leg_config.spot_symbol,
            config.dual_leg_config.stop_loss_threshold,
            config.dual_leg_config.min_profit_threshold
        );
    }
}
